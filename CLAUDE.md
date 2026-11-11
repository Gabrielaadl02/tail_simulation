# tail_sim

MuJoCo simulation of a tendon-driven robotic tail (master's project). A chain of
3D-printed vertebrae joined by universal-joint rings, bent by 4 cables wound on
spools (length / position control, no force sensing on the real hardware).
See `RESEARCH_NOTES.md` for the hardware, the thesis directions, the mechanics
the sim has predicted so far, and the open TODO list.

## Setup on a new machine

The `.venv` is not in git; recreate it. Python 3.15 (tested on 3.15.2, Windows).

```powershell
py -3.15 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If PyYAML fails to build (no 3.15 wheel, no MSVC), install it pure-Python first,
then rerun the line above:

```powershell
$env:PYYAML_FORCE_LIBYAML=0
.\.venv\Scripts\python.exe -m pip install PyYAML==6.0.3 --config-settings="pyyaml_build_config={\"force\":0}"
```

Check it works (headless, no windows):

```powershell
.\.venv\Scripts\python.exe viewer.py --mix 8 0 2 --headless 5 --no-plot
```

## Files

- `config.yaml` - every model parameter (geometry, masses, friction, drive, sim).
  Values marked `PLACEHOLDER` / `FROM DRAWING` are guesses, not measurements.
- `generate_mjcf.py` - builds `tail.xml` (MJCF) from `config.yaml`.
- `tail.xml` - generated; don't hand-edit. `viewer.py` regenerates it every run
  (`--no-regen` to skip).
- `viewer.py` - runs the sim: 3D viewer + Tk "Tendon motors" panel, or
  `--headless SECONDS` which writes `ft_log.csv` and `ft_plot.png`. Also contains
  the tension-dependent stick-slip joint friction logic.
- `mesh_props.py` - volume / COM / inertia of an STL scaled to a measured mass
  (used to fill the inertias in `config.yaml`).
- `meshes/ring.stl`, `meshes/vert.stl` - Onshape exports, in mm.

## Commands

```powershell
.\.venv\Scripts\python.exe generate_mjcf.py                 # config.yaml -> tail.xml
.\.venv\Scripts\python.exe viewer.py --mix UP SIDE TIGHTEN  # interactive (opens windows)
.\.venv\Scripts\python.exe viewer.py --mix 8 0 2 --headless 5   # headless run + CSV/plot
.\.venv\Scripts\python.exe viewer.py --cmd c0 c1 c2 c3 --headless 5  # per-cable commands
.\.venv\Scripts\python.exe mesh_props.py meshes\ring.stl --mass 0.01171223
```

Commands are mm of cable reeled in (`drive: length`, default) or N (`drive: tension`).

## Conventions

- Tail points along +x, gravity -z. Units: m, kg, N, degrees in config.
- Tendons: t0 blue = top/+y, t1 orange = top/-y, t2 aqua = bottom/-y,
  t3 yellow = bottom/+y. Colours are shared by the 3D view, sliders and plots
  (`generate_mjcf.TENDON_HEX`).
- Mix: up = t0+t1, down = t2+t3, +y = t0+t3, -y = t1+t2; Tighten adds to all four.

## Rules for Claude

- **Never open GUI windows when testing** (interactive `viewer.py`, `plt.show()`,
  Tk panels). The user finds pop-ups on their screen disruptive. Test with
  `--headless N`, `MPLBACKEND=Agg`, or `mujoco.Renderer` offscreen. Leave the
  user's own viewer processes alone.
- Always run Python through `.\.venv\Scripts\python.exe`.
- After changing `config.yaml` or `generate_mjcf.py`, regenerate and do a short
  headless run to confirm the model still loads and settles.
- Results from the sim are predictions with guessed parameters; say so when
  reporting them, and don't overwrite measured values in `config.yaml`.
- Git: repo is `https://github.com/Gabrielaadl02/tail_simulation`, branch `main`.
  The user works on more than one computer, so `git pull` before starting.
