## CARLA tools

### Starting CARLA server

#### Recommended (low resource mode):

```bash
./scripts/run/start_carla.sh
```

This starts the CARLA server with predefined settings:

* `-quality-level=Low` — reduces GPU load
* `-RenderOffScreen` — disables on-screen rendering (useful for headless setups)

---

#### Custom configuration

You can pass your own arguments to override the default configuration, for example run:

```bash
./scripts/run/start_carla.sh -ResX=800 -ResY=600 -quality-level=Epic
```

In this case, the provided arguments replace the default ones.

---

### Manual control

You can run the script in two ways.

#### Recommended (wrapper script):

```bash
./scripts/run/start_manual_control.sh
```

The wrapper enables `--sync` mode by default.

---

#### Direct Python execution:

```bash
conda activate carla
cd adapters/carla/
python manual_control.py --sync
```

---

Useful flags:

* `--client-fps` — FPS of the client window (default: 20 FPS)
* `--delta-seconds` — simulation step (default: 0.05 → 20 FPS)
* `--sim-fps` — alternative way to control simulation FPS (overrides `--delta-seconds`)

Example:

```bash
./scripts/run/start_manual_control.sh --client-fps 20 --sim-fps 20
```

Notes:

* It is recommended to keep `client FPS >= simulation FPS` to avoid lag or inconsistent behavior.
* The script is based on the original CARLA `PythonAPI/examples/manual_control.py`, extended with additional arguments for controlling client FPS and simulation time step.

Basic controls:

* `W`, `A`, `S`, `D` — driving
* `Q` — reverse gear
* `P` — toggle autopilot
* `Backspace` — change vehicle
* `C` — change weather
* `ESC` — exit

---

### Traffic generation

You can run the script in two ways.

#### Recommended (wrapper script):

```bash
./scripts/run/start_generate_traffic.sh
```

---

#### Direct Python execution:

```bash
conda activate carla
cd adapters/carla/
python generate_traffic.py
```

---

Useful flags:

* `-n` — number of vehicles
* `-w` — number of walkers

Example:

```bash
./scripts/run/start_generate_traffic.sh -n 20 -w 10
```

Notes:

* The script is based on the original CARLA `PythonAPI/examples/generate_traffic.py` with modifications.
* Walker collision handling has been enabled (by default it is disabled in the original script), which fixes an issue where pedestrians were not properly detected by LiDAR.
