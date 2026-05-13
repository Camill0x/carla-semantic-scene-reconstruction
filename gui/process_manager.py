import json
import os
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, IO, List, Optional, Protocol, TextIO

from gui.catalog import PROCESS_GROUPS, PROCESS_SPECS
from gui.config import LOG_DIR, ROOT_DIR, STATE_PATH


class ProcessHandle(Protocol):
    pid: int

    def poll(self) -> Optional[int]:
        """Return the child-process exit code when it is no longer running."""
        ...

    def wait(self, timeout: Optional[float] = None) -> int:
        """Wait for the child process to exit or until the timeout expires."""
        ...


@dataclass
class ManagedProcess:
    name: str
    command: List[str]
    args: List[str] = field(default_factory=list)
    process: Optional[ProcessHandle] = None
    log_path: Optional[Path] = None
    log_handle: Optional[TextIO] = None
    stdout_handle: Optional[IO[str]] = None
    log_lines: Deque[str] = field(default_factory=lambda: deque(maxlen=2000))
    log_thread: Optional[threading.Thread] = None
    last_exit_code: Optional[int] = None

    def is_running(self) -> bool:
        """Handle is running."""
        return self.process is not None and self.process.poll() is None

    def refresh(self) -> None:
        """Refresh the widget state."""
        if self.process is None:
            return
        exit_code = self.process.poll()
        if exit_code is None:
            return
        self.last_exit_code = int(exit_code)
        self.process = None


class ReattachedProcess:
    def __init__(self, pid: int) -> None:
        """Wrap a running PID so the GUI can reattach to an existing process."""
        self.pid = int(pid)

    def poll(self) -> Optional[int]:
        """Return the exit status of the reattached process when it is no longer alive."""
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return 1
        except PermissionError:
            return None
        return None

    def wait(self, timeout: Optional[float] = None) -> int:
        """Wait for the reattached process to exit or until the timeout expires."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            exit_code = self.poll()
            if exit_code is not None:
                return exit_code
            if deadline is not None and time.monotonic() >= deadline:
                timeout_value = timeout if timeout is not None else 0.0
                raise subprocess.TimeoutExpired(cmd=f"pid {self.pid}", timeout=timeout_value)
            time.sleep(0.1)


class ProjectProcessManager:
    def __init__(self) -> None:
        """Initialize the managed process registry, log files, and persisted GUI state."""
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._log_lock = threading.Lock()
        self.processes: Dict[str, ManagedProcess] = {
            name: ManagedProcess(name=name, command=list(spec.command)) for name, spec in PROCESS_SPECS.items()
        }
        self.ensure_log_files()
        self.load_state()

    def ensure_log_files(self) -> None:
        """Create log files for all managed processes if they do not exist yet."""
        for name, process in self.processes.items():
            log_path = LOG_DIR / f"{name}.log"
            log_path.touch(exist_ok=True)
            process.log_path = log_path

    def pid_is_alive(self, pid: int) -> bool:
        """Return whether a PID still appears to be alive on the system."""
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def refresh_all(self) -> None:
        """Refresh all managed process states and persist the updated GUI state."""
        for process in self.processes.values():
            process.refresh()
        self.save_state()

    def start_process(self, name: str, *, args: Optional[List[str]] = None) -> str:
        """Start one managed process with the provided CLI arguments."""
        process = self.processes[name]
        process.refresh()
        if args is not None:
            process.args = list(args)
        if process.is_running():
            self.save_state()
            return f"{name} is already running"

        log_path = LOG_DIR / f"{name}.log"
        with self._log_lock:
            process.log_lines.clear()
        log_handle = log_path.open("a", encoding="utf-8", buffering=1)
        command = process.command + process.args
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("PYTHONIOENCODING", "utf-8")
        popen = subprocess.Popen(
            command,
            cwd=str(ROOT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            text=True,
            bufsize=1,
            env=env,
        )
        stdout_handle = popen.stdout
        if stdout_handle is None:
            log_handle.close()
            raise RuntimeError(f"Failed to capture stdout for {name}")
        process.process = popen
        process.log_path = log_path
        process.log_handle = log_handle
        process.stdout_handle = stdout_handle
        process.last_exit_code = None
        process.log_thread = threading.Thread(
            target=self._consume_process_output,
            args=(process,),
            name=f"log-reader-{name}",
            daemon=True,
        )
        process.log_thread.start()
        self.save_state()
        return f"Started {name} (pid={popen.pid})"

    def stop_process(self, name: str) -> str:
        """Stop one managed process by escalating through interrupt and termination signals."""
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
        if process.is_running():
            return f"Failed to stop {name}"
        return f"Stopped {name}"

    def restart_process(self, name: str, *, args: Optional[List[str]] = None) -> List[str]:
        """Restart one managed process and return the activity messages."""
        messages = [self.stop_process(name)]
        messages.append(self.start_process(name, args=args))
        return messages

    def stop_many(self, names: List[str]) -> List[str]:
        """Stop multiple managed processes and return their activity messages."""
        return [self.stop_process(name) for name in names]

    def running_process_names(self) -> List[str]:
        """Return the names of all currently running managed processes."""
        self.refresh_all()
        return [name for name, process in self.processes.items() if process.is_running()]

    def process_group_names(self, group_name: str) -> List[str]:
        """Return the process names associated with a configured workflow group."""
        return list(PROCESS_GROUPS[group_name])

    def status_rows(self) -> List[Dict[str, str]]:
        """Build the process-status rows shown in inspector and workflow views."""
        self.refresh_all()
        rows = []
        for name, spec in PROCESS_SPECS.items():
            process = self.processes[name]
            if process.is_running():
                current_process = process.process
                assert current_process is not None
                status = "Running"
                pid = str(current_process.pid)
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
        """Return the available GUI-managed log files by process name."""
        return {name: process.log_path for name, process in self.processes.items() if process.log_path is not None}

    def log_snapshots(self, names: List[str], *, limit_lines: int = 400) -> Dict[str, str]:
        """Return recent log snapshots for the requested process names."""
        snapshots: Dict[str, str] = {}
        for name in names:
            process = self.processes[name]
            with self._log_lock:
                if process.log_lines:
                    snapshots[name] = "".join(list(process.log_lines)[-limit_lines:])
                    continue
            snapshots[name] = self._read_log_tail(process.log_path, limit_lines=limit_lines)
        return snapshots

    def clear_logs(self) -> None:
        """Truncate GUI-managed log files and clear in-memory log buffers."""
        for path in LOG_DIR.glob("*.log"):
            try:
                path.write_text("", encoding="utf-8")
            except Exception:
                continue
        with self._log_lock:
            for process in self.processes.values():
                process.log_lines.clear()

    def summary(self) -> Dict[str, object]:
        """Return aggregate process counts for the GUI summary widgets."""
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
        """Return the persisted CLI arguments for the requested process."""
        return list(self.processes[name].args)

    def save_state(self) -> None:
        """Persist the current GUI-managed process state to disk."""
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
        """Reload persisted GUI-managed process state from disk when possible."""
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

    def _consume_process_output(self, process: ManagedProcess) -> None:
        """Stream one process output into its log file and in-memory buffer."""
        stdout_handle = process.stdout_handle
        log_handle = process.log_handle
        if stdout_handle is None or log_handle is None:
            return
        try:
            for line in stdout_handle:
                log_handle.write(line)
                log_handle.flush()
                with self._log_lock:
                    process.log_lines.append(line)
        finally:
            try:
                stdout_handle.close()
            except Exception:
                pass
            if process.stdout_handle is stdout_handle:
                process.stdout_handle = None
            try:
                log_handle.close()
            finally:
                if process.log_handle is log_handle:
                    process.log_handle = None
            if process.log_thread is threading.current_thread():
                process.log_thread = None

    def _read_log_tail(self, path: Optional[Path], *, limit_lines: int) -> str:
        """Read the tail of a GUI-managed log file."""
        if path is None or not path.exists():
            return ""
        try:
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                return "".join(deque(handle, maxlen=limit_lines))
        except Exception as exc:
            return f"Failed to read log: {exc}"
