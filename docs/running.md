## CARLA tools

### Starting CARLA server

```bash
./scripts/run/start_carla.sh
```

This runs `CarlaUE4.sh` without any extra arguments.

Useful flags:

* `-RenderOffScreen` — runs without the on-screen window, useful for headless or lower-overhead setups.
* `-ResX=... -ResY=...` — changes the render resolution.
* `-quality-level=Low` — lowers visual quality and can reduce GPU load.

Examples:

```bash
./scripts/run/start_carla.sh -RenderOffScreen
./scripts/run/start_carla.sh -ResX=800 -ResY=600 -quality-level=Low
```

Notes:

* Avoid using `-RenderOffScreen` and `-quality-level=Low` together. In CARLA 0.9.15 setup this combination may lead to crashes when changing maps.

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
python -m tools.carla.manual_control --sync
```

---

Useful flags:

* `--map` — map to load before spawning the ego vehicle (default: `Town10HD`)
* `--client-fps` — FPS of the client window (default: 20 FPS)
* `--delta-seconds` — simulation step (default: 0.05 → 20 FPS)
* `--sim-fps` — alternative way to control simulation FPS (overrides `--delta-seconds`)

Example:

```bash
./scripts/run/start_manual_control.sh --map Town05 --client-fps 20 --sim-fps 20
```

Notes:

* It is recommended to keep `client FPS >= simulation FPS` to avoid lag or inconsistent behavior.
* The script is based on the original CARLA `PythonAPI/examples/manual_control.py`, extended with additional arguments for changing the map, controlling client FPS and simulation time step.

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
python -m tools.carla.generate_traffic
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
