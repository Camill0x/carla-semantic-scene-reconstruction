from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
TMP_DIR = ROOT_DIR / "tmp"
LOG_DIR = TMP_DIR / "streaming_gui_logs"
STATE_PATH = TMP_DIR / "streaming_gui_state.json"
APP_NAME = "CARLA Project Control Center"
PROCESS_ORDER = ["manual_control", "producer", "aggregator", "visualizer", "openpcdet", "lanedet"]
