r"""
mesh_props.py - measure an STL: bounding box, volume, centre of mass and
inertia (uniform density, scaled to a measured mass).

    .\.venv\Scripts\python.exe mesh_props.py meshes\ring.stl --mass 0.01171223

Scaling to the real mass handles infill as long as it is roughly uniform.
Values are in the STL's own frame, converted from mm to m.
"""

import argparse
import struct

import numpy as np


def load_stl(path):
    """Return triangles as an (n, 3, 3) array in file units (binary or ASCII STL)."""
    with open(path, "rb") as f:
        data = f.read()
    n = struct.unpack("<I", data[80:84])[0]
    if len(data) == 84 + 50 * n:                      # binary
        rec = np.frombuffer(data[84:], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)),
                                                       ("a", "<u2")]), count=n)
        return rec["v"].astype(np.float64)
    verts = [list(map(float, line.split()[1:4]))      # ASCII
             for line in data.decode(errors="ignore").splitlines()
             if line.strip().startswith("vertex")]
    return np.array(verts).reshape(-1, 3, 3)


def mass_properties(tri):
    """Volume, COM and inertia about the COM of a closed mesh (density 1),
    by summing signed tetrahedra from the origin to each triangle."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    V = vol.sum()
    com = (vol[:, None] * (a + b + c) / 4.0).sum(0) / V
    # Second moments  integral of x_i x_j over each tetrahedron (one vertex at origin).
    S = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            S[i, j] = (vol * (2 * (a[:, i] * a[:, j] + b[:, i] * b[:, j] + c[:, i] * c[:, j])
                              + a[:, i] * b[:, j] + a[:, j] * b[:, i]
                              + a[:, i] * c[:, j] + a[:, j] * c[:, i]
                              + b[:, i] * c[:, j] + b[:, j] * c[:, i]) / 20.0).sum()
    S -= V * np.outer(com, com)                       # move to the COM
    inertia = np.trace(S) * np.eye(3) - S             # density 1
    return V, com, inertia


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stl")
    ap.add_argument("--mass", type=float, help="measured mass [kg]")
    ap.add_argument("--scale", type=float, default=0.001, help="file units -> m (mm = 0.001)")
    args = ap.parse_args()

    tri = load_stl(args.stl) * args.scale
    V, com, I = mass_properties(tri)
    pts = tri.reshape(-1, 3)
    print(f"{args.stl}: {len(tri)} triangles")
    print(f"  bbox min [m] {np.round(pts.min(0), 5)}  max {np.round(pts.max(0), 5)}")
    print(f"  size     [m] {np.round(pts.max(0) - pts.min(0), 5)}")
    print(f"  volume  {V * 1e6:.3f} cm^3   COM [m] {np.round(com, 6)}")
    if args.mass:
        I = I * args.mass / V                         # density = mass / volume
        print(f"  density {args.mass / V:.1f} kg/m^3 (solid PLA ~1240)")
        print(f"  inertia about COM, STL axes [kg m^2]  (Ixx Iyy Izz Ixy Ixz Iyz):")
        print("   ", [float(f"{v:.4g}") for v in (I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2])])


if __name__ == "__main__":
    main()
