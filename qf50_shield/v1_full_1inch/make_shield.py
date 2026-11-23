"""Tungsten-powder shield for the centre of a QF50 five-way cross.

Run with a Python that has CadQuery:   python make_shield.py
Writes STEP + STL for every printed part, an assembly STEP, section drawings
and a ray-traced thickness report next to this file.

Frame = frame of qf50-200-5x.step: arms along +-X, +-Y and +Z, origin at the
centre of the cross, mm.

Parts
  bottom tub  + lid   slides up from below; hugs the lower half of the X/Y arms
  top tub     + lid   lowered over the (open) +Z port; hugs the upper half of
                      the X/Y arms, clears the Z flange with a 76 mm bore
  4 collar quarters + caps (identical)   fill the gap that bore leaves around
                      the Z tube; each slides in along a diagonal between two arms
All tubs are open on their outer flat face: print seam-side down, fill with
tungsten powder, close with the lid.  Four M6 rods clamp the stack.
"""
import math
import os
import sys

import cadquery as cq

# ----------------------------------------------------------------- parameters
CROSS_STEP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qf50-200-5x.step")
OUT = os.path.dirname(os.path.abspath(__file__))

TUBE_R = 25.4        # arm tube outer radius (from the STEP file)
FLANGE_R = 37.465    # QF50 flange outer radius
FLANGE_BACK = 65.02  # distance from centre to the back of each flange

H = 58.0             # shield half-size: reaches +-H along X, Y and up to +H / down to -H in Z
                     # (7 mm short of the flange backs - check against your clamps)
EDGE_CHAMFER = 2.0   # small break on the four vertical edges

CLEAR = 0.6          # radial clearance between print and tube
RB = TUBE_R + CLEAR  # bore radius hugging the arms
RZ = FLANGE_R + 0.55 # bore in the top tub that the Z flange passes through

T_OUT = 2.4          # outer wall and lid thickness
T_IN = 2.0           # wall against the tubes
T_SEAM = 1.2         # wall on the parting surfaces (the two halves back each other up)

STEP_H = 8.0         # the parting surface is at +STEP_H near the centre, -STEP_H outside
STEP_M = 42.0        # half-size of the raised central zone
STEP_GAP = 0.3       # horizontal clearance on the riser between the two zones

BOLT_XY = 52.0       # M6 clamp rods at (+-BOLT_XY, +-BOLT_XY), tucked into the corners
BOLT_R = 3.3
SLEEVE_R = BOLT_R + 2.0

COLLAR_GAP = 0.3     # radial clearance collar <-> top-tub bore
COLLAR_T = 1.6       # collar wall
COLLAR_SPLIT = 0.2   # gap between neighbouring collar quarters
RC = RZ - COLLAR_GAP
COLLAR_Z0 = STEP_H + 0.5

LID_CLEAR = 0.3      # lid locating rim clearance
LID_RIM_W = 1.6
LID_RIM_H = 3.0

BIG = 500.0


# -------------------------------------------------------------------- helpers
def box(x0, x1, y0, y1, z0, z1):
    return cq.Solid.makeBox(x1 - x0, y1 - y0, z1 - z0, cq.Vector(x0, y0, z0))


def cyl(axis, r, a0, a1):
    d = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    p = cq.Vector(*[a0 * c for c in d])
    return cq.Solid.makeCylinder(r, a1 - a0, p, cq.Vector(*d))


def union(*shapes):
    out = shapes[0]
    for s in shapes[1:]:
        out = out.fuse(s)
    return out


def envelope(inset, z0, z1):
    """Square prism |x|,|y| <= H - inset with chamfered vertical edges."""
    a = H - inset
    c = EDGE_CHAMFER - (2 - math.sqrt(2)) * inset
    pts = [(a, a - c), (a - c, a), (-a + c, a), (-a, a - c),
           (-a, -a + c), (-a + c, -a), (a - c, -a), (a, -a + c)]
    return (cq.Workplane("XY").workplane(offset=z0).polyline(pts).close()
            .extrude(z1 - z0).val())


def tube_void(s):
    """Space kept free for the X and Y arms (s=0 outer solid, s=1 cavity)."""
    r = RB + T_IN * s
    zr = STEP_H + T_IN * s
    return union(cyl("x", r, -BIG, BIG), cyl("y", r, -BIG, BIG),
                 box(-BIG, BIG, -r, r, -zr, zr), box(-r, r, -BIG, BIG, -zr, zr))


def bolts(r):
    return union(*[cyl("z", r, -BIG, BIG).translate(cq.Vector(sx * BOLT_XY, sy * BOLT_XY, 0))
                   for sx in (1, -1) for sy in (1, -1)])


def bottom_region(s):
    """Everything below the stepped parting surface."""
    m = STEP_M - T_SEAM * s
    return union(box(-BIG, BIG, -BIG, BIG, -BIG, -STEP_H - T_SEAM * s),
                 box(-m, m, -m, m, -BIG, STEP_H - T_SEAM * s))


def top_region(s):
    """Everything above the stepped parting surface."""
    m = STEP_M + STEP_GAP + T_SEAM * s
    skirt = box(-BIG, BIG, -BIG, BIG, -STEP_H + T_SEAM * s, BIG).cut(box(-m, m, -m, m, -BIG, BIG))
    return union(box(-BIG, BIG, -BIG, BIG, STEP_H + T_SEAM * s, BIG), skirt)


# ---------------------------------------------------------------------- parts
def build():
    parts, powder = {}, {}
    z_lid_b, z_lid_t = -H + T_OUT, H - T_OUT

    # bottom tub
    b_full = envelope(0, -H, H).intersect(bottom_region(0)).cut(tube_void(0)).cut(bolts(BOLT_R))
    b_cav = (envelope(T_OUT, -H - 1, H).intersect(bottom_region(1))
             .cut(tube_void(1)).cut(bolts(SLEEVE_R)))
    parts["bottom_tub"] = b_full.cut(box(-BIG, BIG, -BIG, BIG, -BIG, z_lid_b)).cut(b_cav).clean()
    powder["bottom"] = b_cav.intersect(box(-BIG, BIG, -BIG, BIG, z_lid_b, BIG)).clean()

    # top tub
    zb0, zb1 = cyl("z", RZ, -BIG, BIG), cyl("z", RZ + T_IN, -BIG, BIG)
    t_full = (envelope(0, -H, H).intersect(top_region(0)).cut(tube_void(0))
              .cut(zb0).cut(bolts(BOLT_R)))
    t_cav = (envelope(T_OUT, -H, H + 1).intersect(top_region(1)).cut(tube_void(1))
             .cut(zb1).cut(bolts(SLEEVE_R)))
    parts["top_tub"] = t_full.cut(box(-BIG, BIG, -BIG, BIG, z_lid_t, BIG)).cut(t_cav).clean()
    powder["top"] = t_cav.intersect(box(-BIG, BIG, -BIG, BIG, -BIG, z_lid_t)).clean()

    # lids: plate + locating rim that drops inside the tub wall
    def rim(z0, z1, with_bore):
        ring = envelope(T_OUT + LID_CLEAR, z0, z1).cut(envelope(T_OUT + LID_CLEAR + LID_RIM_W, z0, z1))
        if with_bore:
            r0 = RZ + T_IN + LID_CLEAR
            ring = ring.fuse(cyl("z", r0 + LID_RIM_W, z0, z1).cut(cyl("z", r0, z0, z1)))
        return ring.cut(bolts(SLEEVE_R + LID_CLEAR))

    parts["bottom_lid"] = (envelope(0, -H, z_lid_b).cut(bolts(BOLT_R))
                           .fuse(rim(z_lid_b, z_lid_b + LID_RIM_H, False)).clean())
    parts["top_lid"] = (envelope(0, z_lid_t, H).cut(zb0).cut(bolts(BOLT_R))
                        .fuse(rim(z_lid_t - LID_RIM_H, z_lid_t, True)).clean())

    # Z collar: four identical quarters, split on x = 0 and y = 0
    def collar(s, z1):
        t = COLLAR_T * s
        ring = cyl("z", RC - t, COLLAR_Z0 + t, z1).cut(cyl("z", RB + t, -BIG, BIG))
        return ring.cut(cyl("x", RB + t, -BIG, BIG)).cut(cyl("y", RB + t, -BIG, BIG))

    z_cap = H - COLLAR_T
    g = COLLAR_SPLIT / 2
    quad = box(g, BIG, g, BIG, -BIG, BIG)
    quad_c = box(g + COLLAR_T, BIG, g + COLLAR_T, BIG, -BIG, z_cap)
    q_cav = collar(1, H + 1).intersect(quad_c).clean()
    q_tub = collar(0, z_cap).intersect(quad).cut(q_cav).clean()
    q_cap = cyl("z", RC, z_cap, H).cut(cyl("z", RB, -BIG, BIG)).intersect(quad).clean()
    o, zax = cq.Vector(0, 0, 0), cq.Vector(0, 0, 1)
    for i in range(4):
        parts[f"collar_quarter_{i}"] = q_tub.rotate(o, zax, 90 * i)
        parts[f"collar_cap_{i}"] = q_cap.rotate(o, zax, 90 * i)
        powder[f"collar_{i}"] = q_cav.rotate(o, zax, 90 * i)
    return parts, powder


# --------------------------------------------------------------------- checks
def volume(s):
    return s.Volume() / 1000.0  # cm^3


def solids(shape):
    return shape.Solids() if hasattr(shape, "Solids") else [shape]


def check(parts, powder, cross):
    lines = []
    add = lines.append
    add("PART VOLUMES (cm^3 of plastic)")
    for k, v in parts.items():
        add(f"  {k:22s} {volume(v):8.1f}   solids: {len(solids(v))}")
    add("POWDER VOLUMES (cm^3)")
    tot = 0.0
    for k, v in powder.items():
        tot += volume(v)
        add(f"  {k:22s} {volume(v):8.1f}")
    add(f"  total {tot:.0f} cm^3 -> {tot*9/1000:.1f} / {tot*11/1000:.1f} / {tot*12.5/1000:.1f} kg "
        f"at 9 / 11 / 12.5 g/cm^3 packed density")

    add("INTERFERENCE (mm^3, should all be ~0)")
    names = list(parts)
    for i, a in enumerate(names):
        v = parts[a].intersect(cross).Volume()
        add(f"  {a} x cross: {v:.3f}")
        for b in names[i + 1:]:
            v = parts[a].intersect(parts[b]).Volume()
            if v > 1e-6:
                add(f"  {a} x {b}: {v:.3f}   <-- OVERLAP")

    add("ASSEMBLY MOTION (max overlap with the cross along the path, mm^3)")
    def sweep(name, shapes, direction, dist, others=()):
        worst = 0.0
        for i in range(1, 41):
            d = dist * i / 40
            for s in shapes:
                moved = s.translate(cq.Vector(*[c * d for c in direction]))
                worst = max(worst, moved.intersect(cross).Volume())
                for o in others:
                    worst = max(worst, moved.intersect(o).Volume())
        add(f"  {name}: {worst:.3f}")
    sweep("bottom tub + lid, straight down 150", [parts["bottom_tub"], parts["bottom_lid"]], (0, 0, -1), 150)
    collars = [parts[k] for k in parts if k.startswith("collar")]
    sweep("top tub + lid, straight up 150 (past Z flange and collars)",
          [parts["top_tub"], parts["top_lid"]], (0, 0, 1), 150, others=collars)
    lifted = [parts[k].translate(cq.Vector(0, 0, 5)) for k in ("collar_quarter_0", "collar_cap_0")]
    sweep("collar quarter, lifted 5 then out along the x = y diagonal 120 (top tub off)",
          lifted, (math.sqrt(0.5), math.sqrt(0.5), 0), 120,
          others=[parts["bottom_tub"], parts["collar_quarter_1"], parts["collar_quarter_3"]])
    return lines


def raytrace(powder, n=40000):
    """Path length in powder for rays from the centre of the cross."""
    from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector
    from OCP.gp import gp_Dir, gp_Lin, gp_Pnt

    comp = cq.Compound.makeCompound(list(powder.values()))
    inter = IntCurvesFace_ShapeIntersector()
    inter.Load(comp.wrapped, 1e-7)
    arms = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)]
    port_cos = math.cos(math.atan(TUBE_R / 70.1))  # ray leaves through a port opening
    golden = math.pi * (3 - math.sqrt(5))
    rows = []
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        r = math.sqrt(1 - z * z)
        d = (r * math.cos(golden * i), r * math.sin(golden * i), z)
        inter.Perform(gp_Lin(gp_Pnt(0, 0, 0), gp_Dir(*d)), 0.0, 300.0)
        ws = sorted(inter.WParameter(k) for k in range(1, inter.NbPnt() + 1))
        uniq = []
        for w in ws:
            if not uniq or w - uniq[-1] > 1e-6:
                uniq.append(w)
        if len(uniq) % 2:
            continue  # grazing hit, skip the ray
        path = sum(uniq[k + 1] - uniq[k] for k in range(0, len(uniq), 2))
        cos_arm = max(sum(a * b for a, b in zip(d, arm)) for arm in arms)
        rows.append((d, path, cos_arm, cos_arm > port_cos))
    lines = [f"RAY TRACE from the centre, {len(rows)} directions uniform over the sphere"]
    port = sum(1 for r in rows if r[3]) / len(rows)
    lines.append(f"  {100*port:5.1f} % of directions leave through a port opening (no shield possible)")
    rest = [r for r in rows if not r[3]]
    for thr in (25.4, 22.0, 19.0, 12.7, 6.0, 0.01):
        f = sum(1 for r in rest if r[1] >= thr) / len(rows)
        lines.append(f"  {100*f:5.1f} % see >= {thr:5.2f} mm of powder")
    lines.append("  by angle from the nearest arm axis (median / min powder path, mm):")
    for lo, hi in ((20, 25), (25, 30), (30, 35), (35, 40), (40, 45), (45, 55), (55, 90)):
        sel = sorted(r[1] for r in rest
                     if lo <= math.degrees(math.acos(min(1, r[2]))) < hi)
        if sel:
            lines.append(f"    {lo:2d}-{hi:2d} deg: {sel[len(sel)//2]:5.1f} / {sel[0]:5.1f}   ({100*len(sel)/len(rows):4.1f} % of sphere)")
    down = sorted(r[1] for r in rest if r[0][2] < -0.94)
    lines.append(f"  straight down (within 20 deg of -Z): median {down[len(down)//2]:.1f}, min {down[0]:.1f} mm")
    far = sorted((r for r in rest if r[2] < math.cos(math.radians(40))), key=lambda r: r[1])
    lines.append(f"  more than 40 deg from every arm ({100*len(far)/len(rows):.1f} % of sphere): "
                 f"median {far[len(far)//2][1]:.1f} mm, "
                 f"{100*sum(1 for r in far if r[1] >= 25.4)/len(far):.0f} % >= 25.4, "
                 f"{100*sum(1 for r in far if r[1] >= 19)/len(far):.0f} % >= 19, "
                 f"{100*sum(1 for r in far if r[1] >= 12.7)/len(far):.0f} % >= 12.7")
    lines.append("  thinnest directions more than 40 deg from every arm (azimuth, elevation, mm):")
    for r in far[:12]:
        d = r[0]
        lines.append(f"    az {math.degrees(math.atan2(d[1], d[0])):7.1f}  el {math.degrees(math.asin(d[2])):6.1f}  {r[1]:5.1f}")
    return lines


def min_wall(parts, powder):
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    lines = ["MIN PLASTIC BETWEEN POWDER AND THE OUTSIDE OF ITS OWN TUB (mm)"]
    pairs = (("bottom", "bottom_tub", "bottom_lid"), ("top", "top_tub", "top_lid"),
             ("collar_0", "collar_quarter_0", "collar_cap_0"))
    for pw, tub, lid in pairs:
        closed = parts[tub].fuse(parts[lid]).fuse(powder[pw]).clean()
        outer = cq.Compound.makeCompound(closed.Faces())
        d = BRepExtrema_DistShapeShape(powder[pw].wrapped, outer.wrapped)
        lines.append(f"  {pw:20s} {d.Value():.2f}")
    return lines


# ------------------------------------------------------------------- drawings
def sections(parts, powder, cross):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    views = [("plan at z = 0 (through the arms)", "XY", 0.0),
             ("plan at z = +40 (through the Z collar)", "XY", 40.0),
             ("elevation at y = 0 (through X and Z arms)", "XZ", 0.0),
             ("elevation at y = 34 (through the parting step)", "XZ", 34.0),
             ("diagonal elevation x = y", "DIAG", 0.0)]
    groups = [("cross (steel)", [cross], "#8a8f98"),
              ("tungsten powder", list(powder.values()), "#2b3a55"),
              ("bottom tub + lid", [parts["bottom_tub"], parts["bottom_lid"]], "#d9822b"),
              ("top tub + lid", [parts["top_tub"], parts["top_lid"]], "#3a9a6b"),
              ("Z collar + caps", [v for k, v in parts.items() if k.startswith("collar")], "#b0509a")]

    def slab(kind, off):
        t = 0.02
        if kind == "XY":
            return box(-BIG, BIG, -BIG, BIG, off - t, off + t), (lambda p: (p.x, p.y)), ("x", "y")
        if kind == "XZ":
            return box(-BIG, BIG, off - t, off + t, -BIG, BIG), (lambda p: (p.x, p.z)), ("x", "z")
        s = box(-BIG, BIG, -t, t, -BIG, BIG).rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), 45)
        return s, (lambda p: ((p.x + p.y) / math.sqrt(2), p.z)), ("along x = y", "z")

    fig, axes = plt.subplots(2, 3, figsize=(19, 12.5))
    axes = axes.ravel()
    for ax, (title, kind, off) in zip(axes, views):
        cutter, proj, labels = slab(kind, off)
        for label, shapes, color in groups:
            for shp in shapes:
                sec = shp.intersect(cutter)
                normal_ok = []
                for f in sec.Faces():
                    n = f.normalAt()
                    if kind == "XY" and n.z > 0.9:
                        normal_ok.append(f)
                    elif kind == "XZ" and n.y > 0.9:
                        normal_ok.append(f)
                    elif kind == "DIAG" and (n.y - n.x) / math.sqrt(2) > 0.9:
                        normal_ok.append(f)
                for f in normal_ok:
                    wires = [f.outerWire()] + list(f.innerWires())
                    draw_face(ax, [wire_points(w, proj) for w in wires], color)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(labels[0] + " (mm)"); ax.set_ylabel(labels[1] + " (mm)")
        ax.set_aspect("equal"); ax.set_xlim(-95, 95); ax.set_ylim(-75, 85)
        ax.grid(True, lw=0.3, alpha=0.5)
    ax = axes[-1]
    ax.axis("off")
    for i, (label, _, color) in enumerate(groups):
        ax.add_patch(Polygon([(0.05, 0.85 - 0.12 * i), (0.15, 0.85 - 0.12 * i),
                              (0.15, 0.92 - 0.12 * i), (0.05, 0.92 - 0.12 * i)], color=color))
        ax.text(0.2, 0.87 - 0.12 * i, label, fontsize=14)
    fig.tight_layout()
    path = os.path.join(OUT, "sections.png")
    fig.savefig(path, dpi=110)
    return path


def wire_points(wire, proj):
    """Ordered, discretised outline of a wire as 2-D points."""
    from OCP.BRepAdaptor import BRepAdaptor_CompCurve
    from OCP.GCPnts import GCPnts_QuasiUniformDeflection
    curve = BRepAdaptor_CompCurve(wire.wrapped)
    disc = GCPnts_QuasiUniformDeflection(curve, 0.05)
    return [proj(cq.Vector(disc.Value(i))) for i in range(1, disc.NbPoints() + 1)]


def draw_face(ax, polys, color):
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path
    verts, codes = [], []
    for pts in polys:
        if len(pts) < 3:
            continue
        verts += pts + [pts[0]]
        codes += [Path.MOVETO] + [Path.LINETO] * (len(pts) - 1) + [Path.CLOSEPOLY]
    if verts:
        ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="k", lw=0.4))


# ----------------------------------------------------------------------- main
def main():
    cross = cq.importers.importStep(CROSS_STEP).val()
    parts, powder = build()

    printable = {"bottom_tub": "bottom_tub", "bottom_lid": "bottom_lid", "top_tub": "top_tub",
                 "top_lid": "top_lid", "collar_quarter_0": "collar_quarter_x4",
                 "collar_cap_0": "collar_cap_x4"}
    for key, name in printable.items():
        shp = parts[key]
        cq.exporters.export(cq.Workplane(obj=shp), os.path.join(OUT, f"{name}.step"))
        cq.exporters.export(cq.Workplane(obj=shp), os.path.join(OUT, f"{name}.stl"),
                            tolerance=0.02, angularTolerance=0.1)
    asm = cq.Assembly()
    asm.add(cross, name="cross", color=cq.Color(0.6, 0.6, 0.65))
    colors = {"bottom": cq.Color(0.85, 0.5, 0.17), "top": cq.Color(0.23, 0.6, 0.42),
              "collar": cq.Color(0.7, 0.3, 0.6)}
    for name, shp in parts.items():
        asm.add(shp, name=name, color=colors[name.split("_")[0]])
    for name, shp in powder.items():
        asm.add(shp, name="powder_" + name, color=cq.Color(0.17, 0.23, 0.33))
    asm.save(os.path.join(OUT, "shield_assembly.step"))

    report = check(parts, powder, cross) + min_wall(parts, powder) + raytrace(powder)
    text = "\n".join(report)
    print(text)
    with open(os.path.join(OUT, "report.txt"), "w") as fh:
        fh.write(text + "\n")
    if "--no-sections" not in sys.argv:
        print("sections ->", sections(parts, powder, cross))


if __name__ == "__main__":
    main()
