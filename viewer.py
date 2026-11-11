r"""
viewer.py - run the tail, command the 4 tendon motors, read the base F/T sensor.

Drive mode comes from config.yaml (drive:):
  length  (default)  each command = mm of cable reeled in from the straight tail
                     (negative = paid out). The spool motor is a stiff position
                     servo; cables only pull, go slack, and stall at max_tension.
                     The tension that results is measured and shown.
  tension            each command = cable tension in N.

    # interactive: tighten all 2 mm (pre-tension), then use the panel
    .\.venv\Scripts\python.exe viewer.py --mix 0 0 2

    # bend up 8 mm while tightened 2 mm
    .\.venv\Scripts\python.exe viewer.py --mix 8 0 2

    # no windows at all: simulate 5 s, write ft_log.csv and ft_plot.png
    .\.venv\Scripts\python.exe viewer.py --mix 8 0 2 --headless 5

A "Tendon motors" window opens next to the 3D view:
  MIX   Up/down, Side (+y/-y), Tighten all - move the cables together:
          t0 = +up +side    t1 = +up -side    t2 = -up -side    t3 = -up +side
          and Tighten adds to all four (positive = reel in / more tension)
  TRIM  one slider per cable, added on top of the mix
  Every row: slider (0.01 steps) | - / + (change by --step, default 0.1) | type + Enter.
  The resulting cable commands are shown at the bottom.
Use it rather than the 3D viewer's own Control sliders (those are coarse and
don't update this panel). Closing either window ends the run.

Which tendon is which: each has its own colour, used everywhere (3D view,
slider, plot):
    t0 blue = top/+y    t1 orange = top/-y    t2 aqua = bottom/-y    t3 yellow = bottom/+y
One tendon bends the tail diagonally; use neighbours for pure pitch or yaw:
    up = t0+t1   down = t2+t3   +y = t0+t3   -y = t1+t2
Length drive: reel one side in and the other side out by about the same amount
to bend; reel ALL in a little extra (1-3 mm) to pre-tension and stiffen.

Joint friction grows with cable tension (config joint.friction_per_newton):
tension squeezes the joints, so a pre-tensioned tail locks where it is.

dL = each cable's length change from the straight tail [mm] (negative = shorter).
The model is regenerated from config.yaml on every run (use --no-regen to skip).
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import mujoco
import numpy as np

import generate_mjcf

HERE = os.path.dirname(os.path.abspath(__file__))
FRAME_DT = 1.0 / 60.0      # viewer refresh / logging period [s]
PRINT_DT = 0.5             # console print period [s]
COLORS = generate_mjcf.TENDON_HEX   # same colours as the tendons in the 3D view
COLUMNS = ("t,cmd0,cmd1,cmd2,cmd3,T0_N,T1_N,T2_N,T3_N,Fx,Fy,Fz,Mx,My,Mz,"
           "dL0_mm,dL1_mm,dL2_mm,dL3_mm")
STICK_SPEED = np.radians(0.5)   # a joint slower than this can stick [rad/s]
SLIP_TIME = 0.05                 # after slipping, stay free at least this long [s]
LOCK_FILTER = 0.02               # average the lock torque over ~this long before
                                 #   deciding to slip (ignores numerical spikes) [s]


def is_length_drive(model):
    try:
        return bool(model.numeric("drive_length").data[0])
    except Exception:
        return False


def cmd_unit(model):
    return "mm" if is_length_drive(model) else "N"


def cable_tensions(data):
    """Actual tension in each cable [N]. Tension drive: the commanded force.
    Length drive: what the spool servo is pulling with (0 when slack)."""
    return np.abs(data.actuator_force)


def make_advance(model):
    """Return (advance, c). advance(data) applies tension-dependent stick-slip
    friction at every hinge, then takes one physics step.

    Friction limit per hinge = frictionloss (config) + c * (sum of cable tensions):
    tension squeezes every joint, so a pre-tensioned tail locks.
      * stuck:    the hinge's "lock_" equality holds it at its current angle
      * slipping: if the lock needs more torque than the limit, it lets go and
                  the joint slides with kinetic friction (MuJoCo frictionloss)
      * it sticks again once it has nearly stopped.
    (MuJoCo's own frictionloss alone slowly creeps under a constant load,
    so it can't hold a pose by itself.)"""
    base_friction = model.dof_frictionloss.copy()
    try:
        c = float(model.numeric("friction_per_newton").data[0])
    except Exception:                      # older tail.xml without the value
        c = 0.0
    # (equality id, qpos address, dof address) for every hinge lock
    locks = []
    for i in range(model.neq):
        name = model.equality(i).name
        if name.startswith("lock_"):
            j = model.joint(name[len("lock_"):]).id
            locks.append((i, model.jnt_qposadr[j], model.jnt_dofadr[j]))
    free_until = np.zeros(len(locks))
    avg_torque = np.zeros(model.neq)                   # low-passed lock torque
    alpha = min(1.0, model.opt.timestep / LOCK_FILTER)
    EQ = int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
    started = set()                                    # MjData objects already initialised

    def advance(data):
        if id(data) not in started:
            # Compute forces once before the first step, so the cable tensions
            # (and hence the friction) are known from t = 0.
            mujoco.mj_forward(model, data)
            started.add(id(data))
        extra = c * cable_tensions(data).sum()         # tension-dependent part
        # Torque each active lock applied in the last step, averaged over LOCK_FILTER.
        lock_torque = np.zeros(model.neq)
        rows = data.efc_type[:data.nefc] == EQ
        lock_torque[data.efc_id[:data.nefc][rows]] = np.abs(data.efc_force[:data.nefc][rows])
        avg_torque[:] += (lock_torque - avg_torque) * alpha

        for n, (i, qa, da) in enumerate(locks):
            limit = base_friction[da] + extra
            if data.eq_active[i]:
                if avg_torque[i] > limit:              # load beats static friction -> slip
                    data.eq_active[i] = 0
                    avg_torque[i] = 0.0
                    free_until[n] = data.time + SLIP_TIME
            elif extra > 0 and data.time >= free_until[n] and abs(data.qvel[da]) < STICK_SPEED:
                model.eq_data[i, 0] = data.qpos[qa]    # stick at the current angle
                data.eq_active[i] = 1
            # kinetic friction only while free (a stuck joint is held by its lock)
            model.dof_frictionloss[da] = base_friction[da] + (0.0 if data.eq_active[i] else extra)
        mujoco.mj_step(model, data)

    return advance, c


def tendon_description(model, k):
    """'top/+y' etc., from where tendon k's first hole sits (y, z)."""
    site = model.wrap_objid[model.tendon_adr[k]]
    _, y, z = model.site_pos[site]
    return f"{'top' if z > 0 else 'bottom'}/{'+y' if y > 0 else '-y'}"


def read(model, data):
    """One log row: time, commands, cable tensions (N), base force (N),
    base torque (N*m), cable length change from the straight tail (mm)."""
    dL = (data.ten_length - model.tendon_length0) * 1000
    return np.concatenate(([data.time], data.ctrl, cable_tensions(data),
                           data.sensor("base_force").data,
                           data.sensor("base_torque").data, dL))


def print_row(row, c=0.0, unit="N"):
    t, cmd, T, F, M, dL = row[0], row[1:5], row[5:9], row[9:12], row[12:15], row[15:19]
    cmd_txt = f"cmd=[{cmd[0]:+5.2f} {cmd[1]:+5.2f} {cmd[2]:+5.2f} {cmd[3]:+5.2f}] {unit}  " if unit == "mm" else ""
    print(f"t={t:6.2f}s  {cmd_txt}T=[{T[0]:5.2f} {T[1]:5.2f} {T[2]:5.2f} {T[3]:5.2f}] N  "
          f"fric={c * T.sum():.3f} N*m/joint  "
          f"F=[{F[0]:+6.3f} {F[1]:+6.3f} {F[2]:+6.3f}] N  "
          f"M=[{M[0]:+7.4f} {M[1]:+7.4f} {M[2]:+7.4f}] N*m  "
          f"dL=[{dL[0]:+5.1f} {dL[1]:+5.1f} {dL[2]:+5.1f} {dL[3]:+5.1f}] mm")


MIX_NAMES = ("Up / down", "Side +y / -y", "Tighten all")


def mix_matrix(model):
    """Cable commands from the 3 mixed controls (up, side, tighten).
    Row k = cable k, columns = (up, side toward +y, tighten).
    Cable k sits at hole angle a_k (y = r cos a, z = r sin a), so it gets
        up * sin(a_k) + side * cos(a_k)      (scaled: a diagonal cable gets
    exactly +-up +-side) plus 'tighten' on every cable. Positive = reel in
    (length drive) or more tension (tension drive)."""
    ang = np.array([np.arctan2(model.site_pos[model.wrap_objid[model.tendon_adr[k]]][2],
                               model.site_pos[model.wrap_objid[model.tendon_adr[k]]][1])
                    for k in range(model.ntendon)])
    s = np.max(np.abs(np.sin(ang)))
    M = np.column_stack([np.sin(ang) / s, np.cos(ang) / s, np.ones_like(ang)])
    M[np.abs(M) < 1e-9] = 0.0
    return M


def mixed_command(model, mix, trim):
    """Final cable commands = mix (up, side, tighten) + per-cable trim, clipped."""
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    return np.clip(mix_matrix(model) @ np.asarray(mix, float) + np.asarray(trim, float), lo, hi)


class ControlPanel:
    """Small Tk window for fine motor control (native Tk widgets: fast).
      MIX:  Up/down, Side, Tighten all - move the cables together
      TRIM: one slider per cable, added on top of the mix
    Every row: slider (0.01 steps) | -/+ (change by `step`) | type a value + Enter.
    The resulting cable commands go into data.ctrl and are shown at the bottom.
    Call update() once per frame; `alive` turns False when the window is closed.
    show=False keeps the window hidden (for testing)."""

    def __init__(self, model, data, step, mix0=(0, 0, 0), trim0=(0, 0, 0, 0), show=True):
        import tkinter as tk
        self.model, self.data = model, data
        self.unit = cmd_unit(model)
        self.lo, self.hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
        self.M = mix_matrix(model)
        self.mix = np.array(mix0, float)
        self.trim = np.array(trim0, float)
        self.alive = True

        self.root = root = tk.Tk()
        root.title("Tendon motors")
        root.protocol("WM_DELETE_WINDOW", self.close)
        if not show:
            root.withdraw()
        self.vars = []                       # one DoubleVar per slider row
        span = float(self.hi.max())
        bold = ("Segoe UI", 10, "bold")

        def add_row(row, label, color, vmin, vmax, init, setter):
            var = tk.DoubleVar(value=init)
            tk.Label(root, text=label, fg=color, font=bold, anchor="e", width=14).grid(
                row=row, column=0, padx=4, sticky="e")
            tk.Scale(root, variable=var, from_=vmin, to=vmax, resolution=0.01, digits=5,
                     orient="horizontal", length=320, troughcolor=color, sliderlength=18,
                     showvalue=True).grid(row=row, column=1, padx=4)
            tk.Button(root, text="-", width=3,
                      command=lambda: var.set(max(vmin, round(var.get() - step, 4)))).grid(row=row, column=2)
            tk.Button(root, text="+", width=3,
                      command=lambda: var.set(min(vmax, round(var.get() + step, 4)))).grid(row=row, column=3)
            entry = tk.Entry(root, width=8)
            entry.insert(0, f"{init:.2f}")
            entry.grid(row=row, column=4, padx=4)

            def on_enter(_event):
                try:
                    var.set(float(np.clip(float(entry.get()), vmin, vmax)))
                except ValueError:
                    pass                     # ignore non-numbers

            def on_change(*_):               # slider, buttons, entry and reset all land here
                setter(var.get())
                entry.delete(0, "end")
                entry.insert(0, f"{var.get():.2f}")
                self.apply()

            entry.bind("<Return>", on_enter)
            var.trace_add("write", on_change)
            self.vars.append(var)

        tk.Label(root, text="MIX", font=bold).grid(row=0, column=0, sticky="w", padx=4, pady=(6, 0))
        for j, name in enumerate(MIX_NAMES):
            add_row(1 + j, name, "#404040", -span, span, self.mix[j],
                    lambda v, j=j: self.mix.__setitem__(j, v))
        tk.Label(root, text="TRIM per cable", font=bold).grid(row=4, column=0, sticky="w", padx=4, pady=(10, 0))
        for k in range(model.nu):
            add_row(5 + k, f"t{k} {tendon_description(model, k)}", COLORS[k % len(COLORS)],
                    self.lo[k] - self.hi[k], self.hi[k] - self.lo[k], self.trim[k],
                    lambda v, k=k: self.trim.__setitem__(k, v))

        self.result = tk.StringVar()
        tk.Label(root, textvariable=self.result, font=("Consolas", 10)).grid(
            row=5 + model.nu, column=0, columnspan=5, sticky="w", padx=4, pady=(8, 0))
        # Bottom row in its own frame, so the button keeps a normal fixed size
        # instead of taking the shape of the label column above it.
        bottom = tk.Frame(root)
        bottom.grid(row=6 + model.nu, column=0, columnspan=5, sticky="w", padx=4, pady=6)
        tk.Button(bottom, text="All to 0", width=10, command=self.reset).pack(side="left")
        tk.Label(bottom, text=f"   -/+ step = {step:g} {self.unit}   (type a value + Enter)",
                 fg="#606060").pack(side="left")
        self.apply()

    def apply(self):
        self.data.ctrl[:] = np.clip(self.M @ self.mix + self.trim, self.lo, self.hi)
        self.result.set("cables:  " + "   ".join(f"t{k} {self.data.ctrl[k]:+.2f}"
                                                for k in range(self.model.nu)) + f" {self.unit}")

    def reset(self):
        for var in self.vars:
            var.set(0.0)

    def update(self):
        """Process pending clicks/typing; call once per frame."""
        if self.alive:
            self.root.update()

    def close(self):
        if self.alive:
            self.alive = False
            self.root.destroy()


def run_viewer(model, data, log, step, mix0, trim0):
    import mujoco.viewer
    advance, c = make_advance(model)
    unit = cmd_unit(model)
    steps_per_frame = max(1, round(FRAME_DT / model.opt.timestep))
    next_print = 0.0
    panel = ControlPanel(model, data, step, mix0, trim0)
    with mujoco.viewer.launch_passive(model, data) as v:
        with v.lock():
            v.opt.label = mujoco.mjtLabel.mjLABEL_TENDON   # show t0..t3 names in 3D
        while v.is_running() and panel.alive:
            wall_start = time.perf_counter()
            panel.update()                # handle clicks/typing in the control panel
            for _ in range(steps_per_frame):
                advance(data)
            v.sync()                      # draw the 3D view
            log.append(read(model, data))
            if data.time >= next_print:
                print_row(log[-1], c, unit)
                next_print = data.time + PRINT_DT
            # keep roughly real time
            time.sleep(max(0.0, FRAME_DT - (time.perf_counter() - wall_start)))
    panel.close()


def run_headless(model, data, log, duration):
    advance, c = make_advance(model)
    steps_per_frame = max(1, round(FRAME_DT / model.opt.timestep))
    while data.time < duration:
        for _ in range(steps_per_frame):
            advance(data)
        log.append(read(model, data))
    print_row(log[-1], c, cmd_unit(model))


def save_csv(log, path):
    np.savetxt(path, np.array(log), delimiter=",", header=COLUMNS, comments="", fmt="%.6g")
    print(f"saved {path}")


def plot(log, unit, save_path=None):
    """Plot commands, cable tensions, base F/T and cable length change.
    With save_path: write a PNG, open no window."""
    a = np.array(log)
    t = a[:, 0]
    length = unit == "mm"
    n = 5 if length else 4
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(9, 2.2 * n + 0.6))
    ax_cmd, ax_T = (axes[0], axes[1]) if length else (None, axes[0])
    ax_F, ax_M, ax_dL = axes[-3], axes[-2], axes[-1]
    for k in range(4):
        if length:
            ax_cmd.plot(t, a[:, 1 + k], color=COLORS[k], lw=2, label=f"t{k}")
        ax_T.plot(t, a[:, 5 + k], color=COLORS[k], lw=2, label=f"t{k}")
        ax_dL.plot(t, a[:, 15 + k], color=COLORS[k], lw=2, label=f"t{k}")
    for k, name in enumerate("xyz"):
        ax_F.plot(t, a[:, 9 + k], color=COLORS[k], lw=2, label=f"F{name}")
        ax_M.plot(t, a[:, 12 + k], color=COLORS[k], lw=2, label=f"M{name}")
    labels = (["Reel-in command [mm]"] if length else []) + [
        "Cable tension [N]", "Base force [N]", "Base torque [N·m]",
        "Cable length change [mm]\n(− = shorter)"]
    for ax, ylab in zip(axes, labels):
        ax.set_ylabel(ylab)
        ax.grid(True, color="0.9")
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[-1].set_xlabel("Time [s]")
    axes[0].set_title(f"Tendon tail ({'length' if length else 'tension'} drive): "
                      "base force/torque at site base_ft")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
        print(f"saved {save_path}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mix", type=float, nargs=3, default=[0, 0, 0],
                    metavar=("UP", "SIDE", "TIGHTEN"),
                    help="initial mixed command: up/down, side (+y), tighten all (mm or N)")
    ap.add_argument("--cmd", "--tension", dest="cmd", type=float, nargs=4, default=[0, 0, 0, 0],
                    metavar=("C0", "C1", "C2", "C3"),
                    help="initial per-cable trim, added to the mix: mm reeled in or N")
    ap.add_argument("--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("--model", default=os.path.join(HERE, "tail.xml"))
    ap.add_argument("--no-regen", action="store_true", help="use the existing tail.xml")
    ap.add_argument("--headless", type=float, metavar="SECONDS",
                    help="simulate this long without any window; save ft_log.csv + ft_plot.png")
    ap.add_argument("--step", type=float, default=0.1,
                    help="command change per -/+ click in the control panel (mm or N)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    if not args.no_regen:
        generate_mjcf.generate(args.config, args.model)

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    unit = cmd_unit(model)

    # Initial commands = mix + per-cable trim, clipped to the actuator range.
    data.ctrl[:] = mixed_command(model, args.mix, args.cmd)
    print("Cable commands: " + "  ".join(f"t{k} {data.ctrl[k]:+.2f}" for k in range(model.nu)) + f" {unit}")

    print(f"Drive: {'length (commands = mm reeled in)' if unit == 'mm' else 'tension (commands = N)'}")
    print("Tendons: " + "   ".join(f"t{k} = {tendon_description(model, k)}" for k in range(model.ntendon))
          + "   (colours: blue, orange, aqua, yellow)")
    # Sanity check for the F/T reading: at rest it should read Fz = +M*g
    # (in the base_ft frame, which is aligned with the world here).
    total_mass = model.body_subtreemass[model.body("base").id]
    print(f"Moving + base mass = {total_mass:.4f} kg  -> static support force "
          f"{total_mass * abs(model.opt.gravity[2]):.3f} N")

    log = []
    if args.headless:
        # No windows at all: results go to ft_log.csv and ft_plot.png.
        plt.switch_backend("Agg")
        run_headless(model, data, log, args.headless)
        save_csv(log, os.path.join(HERE, "ft_log.csv"))
        if not args.no_plot:
            plot(log, unit, save_path=os.path.join(HERE, "ft_plot.png"))
    else:
        run_viewer(model, data, log, args.step, args.mix, args.cmd)
        if log and not args.no_plot:
            plot(log, unit)


if __name__ == "__main__":
    main()
