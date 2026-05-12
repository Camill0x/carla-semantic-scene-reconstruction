from typing import Dict, List

from gui.models import FlagSpec, ProcessSpec

PROCESS_SPECS: Dict[str, ProcessSpec] = {
    "server": ProcessSpec(
        name="server",
        title="CARLA Server",
        command=["./carla_server.sh"],
    ),
    "manual_control": ProcessSpec(
        name="manual_control",
        title="Manual Control",
        command=["conda", "run", "-n", "carla_app", "python", "-m", "tools.carla.manual_control"],
    ),
    "traffic": ProcessSpec(
        name="traffic",
        title="Traffic",
        command=["conda", "run", "-n", "carla_app", "python", "-m", "tools.carla.generate_traffic"],
    ),
    "dataset_collect": ProcessSpec(
        name="dataset_collect",
        title="Collect Dataset",
        command=["conda", "run", "-n", "carla_app", "python", "-m", "tools.dataset.collect_dataset"],
    ),
    "dataset_view": ProcessSpec(
        name="dataset_view",
        title="Show Dataset",
        command=["conda", "run", "-n", "carla_app", "python", "-m", "tools.dataset.show_dataset"],
    ),
    "openpcdet_prepare_dataset": ProcessSpec(
        name="openpcdet_prepare_dataset",
        title="Prepare OpenPCDet Dataset",
        command=["conda", "run", "-n", "openpcdet", "python", "-m", "tools.openpcdet.prepare_dataset"],
    ),
    "openpcdet_train": ProcessSpec(
        name="openpcdet_train",
        title="Train OpenPCDet",
        command=["conda", "run", "-n", "openpcdet", "python", "-m", "tools.openpcdet.train"],
    ),
    "openpcdet_test": ProcessSpec(
        name="openpcdet_test",
        title="Test OpenPCDet",
        command=["conda", "run", "-n", "openpcdet", "python", "-m", "tools.openpcdet.test"],
    ),
    "lanedet_prepare_dataset": ProcessSpec(
        name="lanedet_prepare_dataset",
        title="Prepare LaneDet Dataset",
        command=["conda", "run", "-n", "lanedet", "python", "-m", "tools.lanedet.prepare_dataset"],
    ),
    "lanedet_main": ProcessSpec(
        name="lanedet_main",
        title="Run LaneDet",
        command=["conda", "run", "-n", "lanedet", "python", "-m", "tools.lanedet.main"],
    ),
    "benchmark_openpcdet": ProcessSpec(
        name="benchmark_openpcdet",
        title="Benchmark OpenPCDet",
        command=["conda", "run", "-n", "openpcdet", "python", "-m", "tools.benchmark.benchmark_openpcdet"],
    ),
    "benchmark_lanedet": ProcessSpec(
        name="benchmark_lanedet",
        title="Benchmark LaneDet",
        command=["conda", "run", "-n", "lanedet", "python", "-m", "tools.benchmark.benchmark_lanedet"],
    ),
    "benchmark_view_predictions": ProcessSpec(
        name="benchmark_view_predictions",
        title="View Predictions",
        command=["conda", "run", "-n", "carla_app", "python", "-m", "tools.benchmark.view_predictions"],
    ),
    "stream_producer": ProcessSpec(
        name="stream_producer",
        title="Streaming Producer",
        command=["conda", "run", "-n", "carla_app", "python", "-m", "tools.streaming.live_producer"],
    ),
    "stream_aggregator": ProcessSpec(
        name="stream_aggregator",
        title="Streaming Aggregator",
        command=["conda", "run", "-n", "carla_app", "python", "-m", "tools.streaming.live_aggregator"],
    ),
    "stream_visualizer": ProcessSpec(
        name="stream_visualizer",
        title="Streaming Visualizer",
        command=["conda", "run", "-n", "carla_app", "python", "-m", "tools.streaming.live_visualizer"],
    ),
    "stream_openpcdet": ProcessSpec(
        name="stream_openpcdet",
        title="Streaming OpenPCDet",
        command=["conda", "run", "-n", "openpcdet", "python", "-m", "tools.streaming.live_openpcdet_inference"],
    ),
    "stream_lanedet": ProcessSpec(
        name="stream_lanedet",
        title="Streaming LaneDet",
        command=["conda", "run", "-n", "lanedet", "python", "-m", "tools.streaming.live_lanedet_inference"],
    ),
}


PROCESS_GROUPS: Dict[str, List[str]] = {
    "carla": ["server", "manual_control", "traffic"],
    "dataset": ["server", "manual_control", "traffic", "dataset_collect", "dataset_view"],
    "training": [
        "openpcdet_prepare_dataset",
        "openpcdet_train",
        "openpcdet_test",
        "lanedet_prepare_dataset",
        "lanedet_main",
    ],
    "benchmark": ["benchmark_openpcdet", "benchmark_lanedet", "benchmark_view_predictions"],
    "streaming": ["stream_producer", "stream_aggregator", "stream_visualizer", "stream_openpcdet", "stream_lanedet"],
}


MANUAL_CONTROL_FLAGS = [
    FlagSpec(
        "sync",
        "Sync Mode",
        "--sync",
        kind="bool",
        default=True,
        help_text="Synchronous mode is always required for this workflow.",
        enabled=False,
    ),
    FlagSpec("map", "Map", "--map", default="", help_text="Optional CARLA map to load before spawning ego vehicle."),
    FlagSpec(
        "filter",
        "Vehicle Filter",
        "--filter",
        default="vehicle.dodge.charger_2020",
        help_text="Vehicle blueprint filter. Recommended for ego vehicle.",
    ),
    FlagSpec("autopilot", "Autopilot", "--autopilot", kind="bool", default=True),
    FlagSpec("client_fps", "Client FPS", "--client-fps", kind="int", default=20),
    FlagSpec("sim_fps", "Simulation FPS", "--sim-fps", kind="float", default=20.0),
]

TRAFFIC_FLAGS = [
    FlagSpec("vehicles", "Vehicles", "--number-of-vehicles", kind="int", default=20),
    FlagSpec("walkers", "Walkers", "--number-of-walkers", kind="int", default=10),
]

DATASET_COLLECT_FLAGS = [
    FlagSpec("num_frames", "Frames", "--num-frames", kind="int", default=100),
    FlagSpec("every_nth", "Every N-th", "--every-nth", kind="int", default=10),
]

DATASET_VIEW_FLAGS = [
    FlagSpec("run_dir", "Run Directory", "--run-dir", kind="dir", default="", required=True),
    FlagSpec("fps", "Playback FPS", "--fps", kind="float", default=20.0),
    FlagSpec("show_grid", "Show Grid", "--show-grid", kind="bool", default=False),
]

OPENPCDET_PREPARE_FLAGS = [
    FlagSpec("name", "Dataset Name", "--name", default="default"),
    FlagSpec("all_runs", "Use All Runs", "--all", kind="bool", default=True),
    FlagSpec("runs", "Specific Runs", "--runs", default="", help_text="Space-separated run directory names."),
    FlagSpec("val_ratio", "Val Ratio", "--val-ratio", kind="float", default=0.2),
    FlagSpec("test_ratio", "Test Ratio", "--test-ratio", kind="float", default=0.2),
    FlagSpec("seed", "Seed", "--seed", kind="int", default=42),
]

OPENPCDET_TRAIN_FLAGS = [
    FlagSpec(
        "preset",
        "Preset",
        "--preset",
        kind="choice",
        default="cn6-transfusion-ft",
        choices=[
            "cn6-transfusion-ft",
            "cn6-transfusion-zeroshot",
            "cn6-centerpoint-pp-ft",
            "cn6-centerpoint-pp-zeroshot",
        ],
    ),
    FlagSpec("cfg_file", "Config File", "--cfg-file", kind="file", default=""),
    FlagSpec("dataset_name", "Dataset Name", "--dataset-name", default="default"),
    FlagSpec("pretrained_model", "Pretrained Model", "--pretrained-model", kind="file", default=""),
    FlagSpec("ckpt", "Resume Checkpoint", "--ckpt", kind="file", default=""),
    FlagSpec("batch_size", "Batch Size", "--batch-size", kind="int", default=""),
    FlagSpec("epochs", "Epochs", "--epochs", kind="int", default=""),
    FlagSpec("workers", "Workers", "--workers", kind="int", default=4),
    FlagSpec("keep_all_ckpt", "Keep All Checkpoints", "--keep-all-ckpt", kind="bool", default=False),
    FlagSpec("best_metric", "Best Metric", "--best-metric", default="mAP"),
]

OPENPCDET_TEST_FLAGS = [
    FlagSpec(
        "preset",
        "Preset",
        "--preset",
        kind="choice",
        default="cn6-transfusion-ft",
        choices=[
            "cn6-transfusion-ft",
            "cn6-transfusion-zeroshot",
            "cn6-centerpoint-pp-ft",
            "cn6-centerpoint-pp-zeroshot",
        ],
    ),
    FlagSpec("cfg_file", "Config File", "--cfg-file", kind="file", default=""),
    FlagSpec("dataset_name", "Dataset Name", "--dataset-name", default="default"),
    FlagSpec("ckpt", "Checkpoint", "--ckpt", kind="file", default="", required=True),
    FlagSpec("batch_size", "Batch Size", "--batch-size", kind="int", default=""),
    FlagSpec("workers", "Workers", "--workers", kind="int", default=4),
]

LANEDET_PREPARE_FLAGS = [
    FlagSpec("format", "Format", "--format", kind="choice", default="tusimple", choices=["tusimple"], required=True),
    FlagSpec("name", "Dataset Name", "--name", default="default"),
    FlagSpec("all_runs", "Use All Runs", "--all", kind="bool", default=True),
    FlagSpec("runs", "Specific Runs", "--runs", default=""),
    FlagSpec("val_ratio", "Val Ratio", "--val-ratio", kind="float", default=0.2),
    FlagSpec("test_ratio", "Test Ratio", "--test-ratio", kind="float", default=0.2),
    FlagSpec("seed", "Seed", "--seed", kind="int", default=42),
    FlagSpec("max_lanes", "Max Lanes", "--max-lanes", kind="int", default=5),
    FlagSpec("line_width", "Line Width", "--line-width", kind="int", default=15),
]

LANEDET_MAIN_FLAGS = [
    FlagSpec(
        "preset",
        "Preset",
        "--preset",
        kind="choice",
        default="laneatt-r34-tusimple",
        choices=["laneatt-r34-tusimple"],
    ),
    FlagSpec("config", "Config File", "--config", kind="file", default=""),
    FlagSpec("data_root", "Data Root", "--data-root", kind="dir", default=""),
    FlagSpec("validate", "Validation Mode", "--validate", kind="bool", default=False),
    FlagSpec("batch_size", "Batch Size", "--batch-size", kind="int", default=""),
    FlagSpec("epochs", "Epochs", "--epochs", kind="int", default=""),
    FlagSpec("workers", "Workers", "--workers", kind="int", default=""),
    FlagSpec("load_from", "Load From", "--load-from", kind="file", default=""),
    FlagSpec("finetune_from", "Finetune From", "--finetune-from", kind="file", default=""),
    FlagSpec("gpus", "GPUs", "--gpus", default="0", help_text="Space-separated GPU ids."),
    FlagSpec("seed", "Seed", "--seed", kind="int", default=0),
    FlagSpec("view", "Visualization", "--view", kind="bool", default=False),
]

BENCHMARK_OPENPCDET_FLAGS = [
    FlagSpec("run_dir", "Run Directory", "--run-dir", kind="dir", default="", required=True),
    FlagSpec("cfg_file", "Config File", "--cfg-file", kind="file", default="", required=True),
    FlagSpec("ckpt", "Checkpoint", "--ckpt", kind="file", default="", required=True),
    FlagSpec("score_thresh", "Score Threshold", "--score-thresh", kind="float", default=0.05),
    FlagSpec("point_stride", "Point Stride", "--point-stride", kind="int", default=1),
    FlagSpec("warmup", "Warmup Frames", "--warmup", kind="int", default=5),
    FlagSpec("limit", "Frame Limit", "--limit", kind="int", default=""),
    FlagSpec("save_pred", "Save Predictions", "--save-pred", kind="bool", default=False),
]

BENCHMARK_LANEDET_FLAGS = [
    FlagSpec("run_dir", "Run Directory", "--run-dir", kind="dir", default="", required=True),
    FlagSpec("config", "Config File", "--config", kind="file", default="", required=True),
    FlagSpec("ckpt", "Checkpoint", "--ckpt", kind="file", default="", required=True),
    FlagSpec("score_thresh", "Score Threshold", "--score-thresh", kind="float", default=0.2),
    FlagSpec("warmup", "Warmup Frames", "--warmup", kind="int", default=5),
    FlagSpec("limit", "Frame Limit", "--limit", kind="int", default=""),
    FlagSpec("save_pred", "Save Predictions", "--save-pred", kind="bool", default=False),
]

BENCHMARK_VIEW_FLAGS = [
    FlagSpec("run_dir", "Run Directory", "--run-dir", kind="dir", default="", required=True),
    FlagSpec("objects", "Objects Predictions", "--objects", kind="dir", default=""),
    FlagSpec("lanes", "Lanes Predictions", "--lanes", kind="dir", default=""),
    FlagSpec("fps", "Playback FPS", "--fps", kind="float", default=20.0),
    FlagSpec("show_grid", "Show Grid", "--show-grid", kind="bool", default=False),
]

STREAMING_PRODUCER_FLAGS = [
    FlagSpec("every_nth", "Every N-th", "--every-nth", kind="int", default=1),
    FlagSpec("show_grid", "Show Grid", "--show-grid", kind="bool", default=False),
    FlagSpec("verbose", "Verbose", "--verbose", kind="bool", default=False),
]

STREAMING_OPENPCDET_FLAGS = [
    FlagSpec("cfg_file", "Config File", "--cfg-file", kind="file", default="", required=True),
    FlagSpec("ckpt", "Checkpoint", "--ckpt", kind="file", default="", required=True),
    FlagSpec("score_thresh", "Score Threshold", "--score-thresh", kind="float", default=0.05),
    FlagSpec("point_stride", "Point Stride", "--point-stride", kind="int", default=1),
    FlagSpec("verbose", "Verbose", "--verbose", kind="bool", default=False),
]

STREAMING_LANEDET_FLAGS = [
    FlagSpec("config", "Config File", "--config", kind="file", default="", required=True),
    FlagSpec("ckpt", "Checkpoint", "--ckpt", kind="file", default="", required=True),
    FlagSpec("score_thresh", "Score Threshold", "--score-thresh", kind="float", default=0.2),
    FlagSpec("verbose", "Verbose", "--verbose", kind="bool", default=False),
]
