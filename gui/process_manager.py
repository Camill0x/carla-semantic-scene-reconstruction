import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, TextIO

from gui.catalog import PROCESS_GROUPS, PROCESS_SPECS
from gui.config import LOG_DIR, ROOT_DIR, STATE_PATH


@dataclass
class ManagedProcess:
    name: str
    command: List[str]
    args: List[str] = field(default_factory=list)
    process: Optional[subprocess.Popen] = None
    log_path: Optional[Path] = None
    log_handle: Optional[TextIO] = None
    last_exit_code: Optional[int] = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def refresh(self) -> None:
        if self.process is None:
            return
        exit_code = self.process.poll()
        if exit_code is None:
            return
        self.last_exit_code = int(exit_code)
        self.process = None
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None


class ReattachedProcess:
    def __init__(self, pid: int) -> None:
        self.pid = int(pid)

    def poll(self) -> Optional[int]:
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return 1
        except PermissionError:
            return None
        return None

    def wait(self, timeout: Optional[float] = None) -> int:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            exit_code = self.poll()
            if exit_code is not None:
                return exit_code
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired(cmd=f"pid {self.pid}", timeout=timeout)
            time.sleep(0.1)


class ProjectProcessManager:
    def __init__(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.processes: Dict[str, ManagedProcess] = {
            name: ManagedProcess(name=name, command=list(spec.command)) for name, spec in PROCESS_SPECS.items()
        }
        self.ensure_log_files()
        self.load_state()

    def ensure_log_files(self) -> None:
        for name, process in self.processes.items():
            log_path = LOG_DIR / f"{name}.log"
            log_path.touch(exist_ok=True)
            process.log_path = log_path

    def pid_is_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def refresh_all(self) -> None:
        for process in self.processes.values():
            process.refresh()
        self.save_state()

    def start_process(self, name: str, *, args: Optional[List[str]] = None) -> str:
        process = self.processes[name]
        process.refresh()
        if args is not None:
            process.args = list(args)
        if process.is_running():
            self.save_state()
            return f"{name} is already running"

        log_path = LOG_DIR / f"{name}.log"
        log_handle = log_path.open("a", encoding="utf-8")
        command = process.command + process.args
        popen = subprocess.Popen(
            command,
            cwd=str(ROOT_DIR),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            text=True,
        )
        process.process = popen
        process.log_path = log_path
        process.log_handle = log_handle
        process.last_exit_code = None
        self.save_state()
        return f"Started {name} (pid={popen.pid})"

    def stop_process(self, name: str) -> str:
        process = self.processes[name]
        process.refresh()
        if process.process is None:
            return f"{name} is not running"

        for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]:
            try:
                os.killpg(process.process.pid, sig)
            except ProcessLookupError:
                break
            try:
                process.process.wait(timeout=5.0)
                break
            except subprocess.TimeoutExpired:
                continue

        process.refresh()
        self.save_state()
        return f"Stopped {name}"

    def restart_process(self, name: str, *, args: Optional[List[str]] = None) -> List[str]:
        messages = [self.stop_process(name)]
        messages.append(self.start_process(name, args=args))
        return messages

    def stop_many(self, names: List[str]) -> List[str]:
        return [self.stop_process(name) for name in names]

    def running_process_names(self) -> List[str]:
        self.refresh_all()
        return [name for name, process in self.processes.items() if process.is_running()]

    def process_group_names(self, group_name: str) -> List[str]:
        return list(PROCESS_GROUPS[group_name])

    def status_rows(self) -> List[Dict[str, str]]:
        self.refresh_all()
        rows = []
        for name, spec in PROCESS_SPECS.items():
            process = self.processes[name]
            if process.is_running():
                status = "Running"
                pid = str(process.process.pid)
            elif process.last_exit_code is not None:
                status = f"Closed ({process.last_exit_code})"
                pid = "-"
            else:
                status = "Idle"
                pid = "-"

            rows.append(
                {
                    "name": name,
                    "title": spec.title,
                    "status": status,
                    "pid": pid,
                    "args": " ".join(process.args),
                    "last_exit": "-" if process.last_exit_code is None else str(process.last_exit_code),
                    "log": str(process.log_path) if process.log_path is not None else "",
                }
            )
        return rows

    def available_log_files(self) -> Dict[str, Path]:
        return {name: process.log_path for name, process in self.processes.items() if process.log_path is not None}

    def clear_logs(self) -> None:
        for path in LOG_DIR.glob("*.log"):
            try:
                path.write_text("", encoding="utf-8")
            except Exception:
                continue

    def summary(self) -> Dict[str, object]:
        self.refresh_all()
        running = [name for name, process in self.processes.items() if process.is_running()]
        streaming = PROCESS_GROUPS["streaming"]
        return {
            "running_count": len(running),
            "total_count": len(self.processes),
            "running_names": running,
            "streaming_count": sum(1 for name in streaming if self.processes[name].is_running()),
        }

    def args_for(self, name: str) -> List[str]:
        return list(self.processes[name].args)

    def save_state(self) -> None:
        state = {}
        for name, process in self.processes.items():
            process.refresh()
            pid = process.process.pid if process.process is not None and process.process.poll() is None else None
            state[name] = {
                "args": list(process.args),
                "pid": pid,
                "log_path": str(process.log_path) if process.log_path is not None else None,
                "last_exit_code": process.last_exit_code,
            }
        with STATE_PATH.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)

    def load_state(self) -> None:
        if not STATE_PATH.exists():
            return
        try:
            state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(state, dict):
            return
        for name, process in self.processes.items():
            raw = state.get(name)
            if not isinstance(raw, dict):
                continue
            args = raw.get("args")
            if isinstance(args, list):
                process.args = [str(item) for item in args]
            log_path = raw.get("log_path")
            if isinstance(log_path, str) and log_path:
                process.log_path = Path(log_path)
            last_exit_code = raw.get("last_exit_code")
            if isinstance(last_exit_code, int):
                process.last_exit_code = last_exit_code
            pid = raw.get("pid")
            if isinstance(pid, int) and self.pid_is_alive(pid):
                process.process = ReattachedProcess(pid)
