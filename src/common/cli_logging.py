from datetime import datetime


def format_verbose_log(component: str, message: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"{timestamp} | {component} | {message}"


def print_verbose(enabled: bool, component: str, message: str) -> None:
    if enabled:
        print(format_verbose_log(component, message))
