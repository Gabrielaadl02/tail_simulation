r"""
generate_mjcf.py - build the MJCF model (tail.xml) of the tendon-driven tail
from config.yaml.

    .\.venv\Scripts\python.exe generate_mjcf.py                 # config.yaml -> tail.xml
    .\.venv\Scripts\python.exe generate_mjcf.py -c my.yaml -o my.xml

Model structure (body tree). Each universal joint is a ring (Part 5) that the
forks of two neighbouring links pin onto, at 90 deg to each other:

    world
     └ base                 fixed, F/T sensor site "base_ft" at its origin
        └ ring_v1           pitch hinge "v1_pitch" (about y) - pinned to base fork
           └ v1             yaw hinge "v1_yaw" (about z), same point - vertebra rear fork
              └ ring_v2     pitch hinge "v2_pitch" - pinned to v1's front fork
                 └ v2
                    └ ...
                       └ ring_end   end_with: ring  -> pitch hinge "end_pitch" only;
                                    the tendons are tied off in its holes
                         (end_with: tip -> ring_tip + tip body, tendons end in the tip)

Tendon k is a spatial tendon threaded through a site at every hole:
base hole -> ring holes -> ... -> tip anchor. Between sites it is a straight
line, which is what a cable through holes does. A motor actuator applies a
pure tension to each tendon.
"""

import argparse
import math
import os
import xml.etree.ElementTree as ET

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))

# Colours (RGBA)
BONE = "0.85 0.85 0.80 1"
BASE = "0.35 0.35 0.40 1"
HOLE = "1.0 0.6 0.0 1"
# One colour per tendon t0..t3; viewer.py uses the same colours for its
# sliders and plot lines, so you can match them by eye.
TENDON_HEX = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # blue, orange, aqua, yellow


def hex_rgba(h):
    return vec([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)] + [1])


def vec(values):
    """Format numbers as an MJCF attribute string, e.g. [1, 0, 0] -> '1 0 0'."""
    return " ".join(f"{float(v):.6g}" for v in values)


def add_inertial(body, part):
    """Explicit mass and inertia, so MuJoCo never computes them from the mesh."""
    inertia = part["inertia"]
    attrs = {"pos": vec(part["com"]), "mass": f"{float(part['mass']):.6g}"}
    if len(inertia) == 3:
        attrs["diaginertia"] = vec(inertia)
    elif len(inertia) == 6:
        attrs["fullinertia"] = vec(inertia)  # Ixx Iyy Izz Ixy Ixz Iyz
    else:
        raise ValueError("inertia needs 3 (diagonal) or 6 (full) values")
    ET.SubElement(body, "inertial", attrs)


def add_visual(body, part, mesh_name, x0, x1, radius, rgba=BONE):
    """Visual geom: the STL if available, otherwise a placeholder cylinder from
    x0 to x1. contype/conaffinity = 0 -> this geom never collides with anything."""
    no_collision = {"contype": "0", "conaffinity": "0", "rgba": rgba}
    if mesh_name:
        euler = [math.radians(a) for a in part.get("mesh_euler_deg", [0, 0, 0])]
        ET.SubElement(body, "geom", {
            "type": "mesh", "mesh": mesh_name,
            "pos": vec(part.get("mesh_pos", [0, 0, 0])), "euler": vec(euler),
            **no_collision})
    else:
        ET.SubElement(body, "geom", {
            "type": "cylinder", "size": f"{radius:.6g}",
            "fromto": vec([x0, 0, 0, x1, 0, 0]), **no_collision})


HINGES = []    # names of all hinges, in order (used for the friction locks)


def add_hinge(body, name, axis, range_deg, jcfg):
    """Hinge through the body origin, with a joint-limit stop and optional
    viscous damping / dry friction (frictionloss) from the config."""
    HINGES.append(name)
    lim = math.radians(range_deg)
    ET.SubElement(body, "joint", {
        "name": name, "type": "hinge", "pos": "0 0 0",
        "axis": axis, "limited": "true", "range": vec([-lim, lim]),
        "damping": f"{jcfg['damping']:.6g}", "armature": f"{jcfg['armature']:.6g}",
        "frictionloss": f"{jcfg.get('frictionloss', 0.0):.6g}",
        "solreflimit": vec(jcfg["limit_solref"]), "solimplimit": vec(jcfg["limit_solimp"])})


def add_hole_sites(body, name, hole_x, radius, angles_deg, paths):
    """One site per tendon hole. Appends site names to each tendon's path."""
    for k, ang in enumerate(angles_deg):
        a = math.radians(ang)
        y, z = radius * math.cos(a), radius * math.sin(a)
        for j, x in enumerate(hole_x):
            site = f"{name}_hole{k}_{j}"
            ET.SubElement(body, "site", {"name": site, "pos": vec([x, y, z]),
                                         "size": "0.001", "rgba": HOLE})
            paths[k].append(site)


def find_mesh(cfg, cfg_dir, key):
    """Return the STL path if it exists, else None (-> placeholder geometry)."""
    path = os.path.join(cfg_dir, cfg["mesh_dir"], cfg[key]["mesh"])
    if os.path.isfile(path):
        return path
    print(f"[generate_mjcf] {path} not found -> using a placeholder cylinder for '{key}'")
    return None


def build(cfg, cfg_dir, out_path):
    n = int(cfg["n_vertebrae"])
    s = float(cfg["spacing"])
    r = float(cfg["hole_radius"])
    angles = cfg["tendon_angles_deg"]
    jcfg, base, vert, tip = cfg["joint"], cfg["base"], cfg["vertebra"], cfg["tip"]
    ring = cfg["ring"]
    sim, tcfg = cfg["sim"], cfg["tendon"]
    total_len = base["joint_x"] + n * s + (tip["length"] if cfg.get("end_with", "ring") == "tip" else 0)

    HINGES.clear()
    root = ET.Element("mujoco", {"model": "tendon_tail"})

    # inertiafromgeom=false: use only the <inertial> values from the config.
    out_dir = os.path.dirname(os.path.abspath(out_path))
    mesh_dir = os.path.relpath(os.path.join(cfg_dir, cfg["mesh_dir"]), out_dir)
    ET.SubElement(root, "compiler", {"angle": "radian", "inertiafromgeom": "false",
                                     "meshdir": mesh_dir.replace("\\", "/")})
    ET.SubElement(root, "option", {"timestep": f"{sim['timestep']:.6g}",
                                   "gravity": vec(sim["gravity"]),
                                   "integrator": "implicitfast"})
    # Frame the free camera on the tail.
    ET.SubElement(root, "statistic", {
        "center": vec([sim["base_pos"][0] + total_len / 2, sim["base_pos"][1], sim["base_pos"][2]]),
        "extent": f"{total_len:.6g}"})

    # Meshes (only the ones that exist on disk).
    asset = ET.SubElement(root, "asset")
    mesh_names = {}
    for key in ("vertebra", "ring") + (("tip",) if cfg.get("end_with", "ring") == "tip" else ()):
        path = find_mesh(cfg, cfg_dir, key)
        if path:
            mesh_names[key] = f"{key}_mesh"
            sc = cfg["mesh_scale"]
            ET.SubElement(asset, "mesh", {"name": mesh_names[key], "file": cfg[key]["mesh"],
                                          "scale": vec([sc, sc, sc])})

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", {"pos": "0 0 2", "dir": "0 0 -1", "directional": "true"})
    # Reference floor, visual only.
    ET.SubElement(world, "geom", {"name": "floor", "type": "plane", "size": "1 1 0.01",
                                  "rgba": "0.9 0.9 0.9 1", "contype": "0", "conaffinity": "0"})

    paths = [[] for _ in angles]              # site names along each tendon

    # --- base (no joint = welded to the world) ---
    b = ET.SubElement(world, "body", {"name": "base", "pos": vec(sim["base_pos"])})
    add_inertial(b, base)
    ET.SubElement(b, "geom", {"type": "box", "pos": vec(base["box_pos"]),
                              "size": vec(base["box_half_size"]), "rgba": BASE,
                              "contype": "0", "conaffinity": "0"})
    ET.SubElement(b, "site", {"name": "base_ft", "pos": "0 0 0", "size": "0.004",
                              "rgba": "0 0.6 1 1"})
    add_hole_sites(b, "base", base["hole_x"], r, angles, paths)

    # --- joints + links: ring, then vertebra (or tip), repeated ---
    half_t = ring["thickness"] / 2

    def add_ring(parent, name, pos, locked=False):
        """Ring pinned to the previous link's front fork -> pitch hinge about y.
        locked=True: no hinge, the ring is rigidly fixed to its parent."""
        rg = ET.SubElement(parent, "body", {"name": f"ring_{name}", "pos": vec(pos)})
        if not locked:
            add_hinge(rg, f"{name}_pitch", "0 1 0", jcfg["pitch_range_deg"], jcfg)
        add_inertial(rg, ring)
        add_visual(rg, ring, mesh_names.get("ring"), -half_t, half_t,
                   ring["outer_radius"], rgba="0.4 0.6 0.9 0.5")
        add_hole_sites(rg, f"ring_{name}", ring["hole_x"], r, angles, paths)
        return rg

    end_with = cfg.get("end_with", "ring")
    links = [f"v{i}" for i in range(1, n + 1)] + (["tip"] if end_with == "tip" else [])
    parent, pos = b, [base["joint_x"], 0, 0]
    for name in links:
        part = tip if name == "tip" else vert
        length = tip["length"] if name == "tip" else s
        # The first ring can be locked to the base (config: lock_first_ring).
        rg = add_ring(parent, name, pos, locked=(name == "v1" and cfg.get("lock_first_ring", False)))

        # Link: its rear fork is pinned to the ring -> yaw hinge about z, same point.
        v = ET.SubElement(rg, "body", {"name": name, "pos": "0 0 0"})
        add_hinge(v, f"{name}_yaw", "0 0 1", jcfg["yaw_range_deg"], jcfg)
        add_inertial(v, part)
        add_visual(v, part, mesh_names.get("tip" if name == "tip" else "vertebra"),
                   half_t, length - half_t, 0.2 * r)
        add_hole_sites(v, name, part["hole_x"], r, angles, paths)
        parent, pos = v, [s, 0, 0]

    if end_with == "ring":
        # The last ring is the tip: pinned to vN's front fork (pitch hinge only),
        # and the tendons are tied off in its holes.
        # lock_end_ring: true -> ring fixed to vN (no end_pitch hinge).
        parent = add_ring(parent, "end", pos, locked=cfg.get("lock_end_ring", False))
        tip_marker = [0, 0, 0]                        # ring centre
    else:
        tip_marker = [tip["length"], 0, 0]            # end of the tip part
    ET.SubElement(parent, "site", {"name": "tip_end", "pos": vec(tip_marker),
                                   "size": "0.003", "rgba": "0 0.8 0 1"})

    drive = cfg.get("drive", "length")
    if drive not in ("length", "tension"):
        raise ValueError("drive must be 'length' or 'tension'")

    # --- tendons: straight segments between consecutive hole sites ---
    tendon = ET.SubElement(root, "tendon")
    for k, sites in enumerate(paths):
        attrs = {"name": f"t{k}", "width": f"{tcfg['width']:.6g}",
                 "rgba": hex_rgba(TENDON_HEX[k % len(TENDON_HEX)])}
        sp = ET.SubElement(tendon, "spatial", attrs)
        for site in sites:
            ET.SubElement(sp, "site", {"site": site})

    act = ET.SubElement(root, "actuator")
    for k in range(len(paths)):
        if drive == "tension":
            # Tension actuator. gear=-1: a positive control (newtons) pulls, i.e.
            # shortens the tendon. ctrlrange starts at 0, so it can never push.
            ET.SubElement(act, "motor", {"name": f"T{k}", "tendon": f"t{k}", "gear": "-1",
                                         "ctrllimited": "true",
                                         "ctrlrange": vec([0, tcfg["max_tension"]])})
        else:
            # Length drive = spool motor as a stiff, force-limited position servo
            # on the cable. Control = mm of cable reeled in (negative = paid out),
            # relative to the straight tail. Force on the tendon:
            #     F = kp * (L_target - L) - kv * dL/dt,  L_target = L0 - reel
            # clamped to [-max_tension, 0]: it only pulls (a cable can't push), it
            # goes slack when the cable is shorter than commanded, and the motor
            # stalls at max_tension. Written as MuJoCo's affine actuator:
            #     F = gain*ctrl + b0 + b1*L + b2*dL/dt
            #     gain = -kp/1000 (ctrl in mm), b0 = kp*L0, b1 = -kp, b2 = -kv
            # L0 (straight length) is only known after compiling, so b0 is filled
            # in by a second pass below.
            kp, kv = float(tcfg["cable_stiffness"]), float(tcfg["cable_damping"])
            reel = float(tcfg.get("max_reel_mm", 40.0))
            ET.SubElement(act, "general", {"name": f"L{k}", "tendon": f"t{k}",
                                           "ctrllimited": "true", "ctrlrange": vec([-reel, reel]),
                                           "forcelimited": "true",
                                           "forcerange": vec([-tcfg["max_tension"], 0]),
                                           "gaintype": "fixed", "gainprm": f"{-kp / 1000:.6g}",
                                           "biastype": "affine",
                                           "biasprm": vec([0, -kp, -kv])})

    # --- friction "locks": one per hinge, off by default ---
    # viewer.py switches a lock on to hold a joint at its current angle when it
    # has (nearly) stopped and the friction can hold it (static friction), and
    # off again when the torque needed exceeds the friction limit (slip).
    eq = ET.SubElement(root, "equality")
    for h in HINGES:
        ET.SubElement(eq, "joint", {"name": f"lock_{h}", "joint1": h, "polycoef": "0 0 0 0 0",
                                    "active": "false",
                                    "solref": vec(jcfg.get("stick_solref", [0.001, 1.0])),
                                    "solimp": vec(jcfg.get("stick_solimp", [0.99, 0.999, 0.001]))})

    # --- tension-dependent friction coefficient, read by viewer.py at runtime ---
    custom = ET.SubElement(root, "custom")
    ET.SubElement(custom, "numeric", {"name": "friction_per_newton",
                                      "data": f"{float(jcfg.get('friction_per_newton', 0.0)):.6g}"})
    # 1 = length-driven cables, 0 = tension-driven (read by viewer.py)
    ET.SubElement(custom, "numeric", {"name": "drive_length",
                                      "data": "1" if drive == "length" else "0"})

    # --- sensors ---
    # force/torque: what the world exerts on the base (base + whole tail), at
    # site base_ft, expressed in the base_ft frame.
    sens = ET.SubElement(root, "sensor")
    ET.SubElement(sens, "force", {"name": "base_force", "site": "base_ft"})
    ET.SubElement(sens, "torque", {"name": "base_torque", "site": "base_ft"})
    ET.SubElement(sens, "framepos", {"name": "tip_pos", "objtype": "site", "objname": "tip_end"})
    for k in range(len(paths)):
        ET.SubElement(sens, "tendonpos", {"name": f"t{k}_length", "tendon": f"t{k}"})

    ET.indent(root)
    ET.ElementTree(root).write(out_path, encoding="utf-8", xml_declaration=False)

    if drive == "length":
        # Second pass: compile once to get each tendon's straight length L0,
        # then put kp*L0 into the servo's bias so ctrl = 0 means "straight length".
        import mujoco
        model = mujoco.MjModel.from_xml_path(out_path)
        kp = float(tcfg["cable_stiffness"])
        for k, motor in enumerate(act.findall("general")):
            L0 = float(model.tendon_length0[model.tendon(f"t{k}").id])
            motor.set("biasprm", vec([kp * L0, -kp, -float(tcfg["cable_damping"])]))
        ET.ElementTree(root).write(out_path, encoding="utf-8", xml_declaration=False)
    n_hinges = (len(links) * 2 - bool(cfg.get("lock_first_ring", False))
                + (end_with == "ring" and not cfg.get("lock_end_ring", False)))
    print(f"[generate_mjcf] wrote {out_path}  ({n} vertebrae, ends with {end_with}, "
          f"{n_hinges} hinges, {drive}-driven)")
    return out_path


def generate(config_path=os.path.join(HERE, "config.yaml"), out_path=os.path.join(HERE, "tail.xml")):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return build(cfg, os.path.dirname(os.path.abspath(config_path)), out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", default=os.path.join(HERE, "config.yaml"))
    ap.add_argument("-o", "--out", default=os.path.join(HERE, "tail.xml"))
    args = ap.parse_args()
    generate(args.config, args.out)
