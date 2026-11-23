"""Compact tungsten-powder shield for the centre of a QF50 five-way cross.

Target: ~0.5 inch of powder around the junction, 1-2 kg of tungsten in total.
Run with a Python that has CadQuery:   python make_shield_compact.py

Frame = frame of qf50-200-5x.step: arms along +-X, +-Y and +Z, origin at the
centre of the cross, mm.

Parts (all open tubs: print, fill with powder, glue the lid on)
  bottom tub + lid    cup under the junction with four corner towers; U-slots
                      for the four horizontal arms, so it goes straight up
                      from below
  2 X blocks + lids   sit on top of the +-X arms and wrap the Z tube
  2 Y blocks + lids   sit on top of the +-Y arms
Each block is set on its arm outboard of the shield and slid in towards the
centre; the bottom tub then comes up between them.  Two straps (cable ties or
hose clamps) around the outside, one over each X block and under the tub, hang
the tub from the blocks.
"""
import math
import os
import sys

import cadquery as cq

CROSS_STEP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qf50-200-5x.step")
OUT = os.path.dirname(os.path.abspath(__file__))

TUBE_R = 25.4
RB = TUBE_R + 0.6     # bore radius hugging the tubes
T = 1.6               # every printed wall
THICK = 12.7          # powder layer straight down / straight out from a tube

E = 40.0              # shield reaches +-E along X and Y and up to +E in Z
ZB = RB + T + THICK + T   # depth below centre (41.9)
CORNER_CUT = 97.0     # bottom corners removed where |x| + |y| - z > CORNER_CUT

GAP = 0.25            # clearance between block and tower faces
X_EAR_MIN = 10.0      # X blocks wrap the Z tube back to |x| = X_EAR_MIN
BLOCK_Z0 = 8.0        # blocks' side tongues stop this far above the arm axis

LID_CLEAR, LID_RIM_W, LID_RIM_H = 0.25, 1.2, 2.5
BIG = 400.0


def box(x0, x1, y0, y1, z0, z1):
    return cq.Solid.makeBox(x1 - x0, y1 - y0, z1 - z0, cq.Vector(x0, y0, z0))


def cyl(axis, r, a0=-BIG, a1=BIG, at=(0, 0, 0)):
    d = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    p = cq.Vector(*[a0 * c + o for c, o in zip(d, at)])
    return cq.Solid.makeCylinder(r, a1 - a0, p, cq.Vector(*d))


def union(*shapes):
    out = shapes[0]
    for s in shapes[1:]:
        out = out.fuse(s)
    return out


def envelope(t, open_bottom=False, open_top=False):
    """Outer block inset by t: box with the four bottom corners cut off.
    open_* leaves that face where it is (the tub is open there, closed by a lid)."""
    e = box(-E + t, E - t, -E + t, E - t,
            -ZB - 1 if open_bottom else -ZB + t, E + 1 if open_top else E - t)
    c = CORNER_CUT - t * math.sqrt(3)
    n = 1 / math.sqrt(3)
    for sx in (1, -1):
        for sy in (1, -1):
            # half-space sx*x + sy*y - z > c, as a big box rotated onto the plane
            cutter = box(-BIG, BIG, -BIG, BIG, 0, BIG)
            normal = cq.Vector(sx * n, sy * n, -n)
            axis = cq.Vector(0, 0, 1).cross(normal)
            ang = math.degrees(math.acos(-n))
            cutter = cutter.rotate(cq.Vector(0, 0, 0), axis, ang).translate(normal * (c * n))
            e = e.cut(cutter)
    return e


def bottom_region(t, opened=False):
    """Bottom tub (t = 0) or the same shape with every wall moved in by t."""
    r = RB + t
    slots = union(cyl("x", r), cyl("y", r),
                  box(-BIG, BIG, -r, r, 0, BIG), box(-r, r, -BIG, BIG, 0, BIG))
    return envelope(t, open_bottom=opened).cut(slots)


def block_region(t, kind, opened=False):
    """Saddle block on the +X ('x') or +Y ('y') arm, walls moved in by t."""
    w = RB - GAP - t
    if kind == "x":
        b = box(X_EAR_MIN + t, BIG, -w, w, BLOCK_Z0 + t, BIG)
    else:
        b = box(-w, w, RB + GAP + t, BIG, BLOCK_Z0 + t, BIG)
    r = RB + t
    return (envelope(t, open_top=opened).intersect(b)
            .cut(cyl("x", r)).cut(cyl("y", r)).cut(cyl("z", r)))


def tub_and_lid(region, z_lid, lid_above):
    """Split region(0) at the lid plane; cavity = region(T) open through the lid face;
    the lid gets a locating rim that drops inside the tub wall."""
    slab = (lambda a, b: box(-BIG, BIG, -BIG, BIG, a, b))
    lid_side = slab(z_lid, BIG) if lid_above else slab(-BIG, z_lid)
    outer = region(0)
    cav = region(T, True)
    tub = outer.cut(lid_side).cut(cav).clean()
    powder = cav.cut(lid_side).clean()
    rim_slab = slab(z_lid - LID_RIM_H, z_lid) if lid_above else slab(z_lid, z_lid + LID_RIM_H)
    rim = (region(T + LID_CLEAR, True).cut(region(T + LID_CLEAR + LID_RIM_W, True))
           .intersect(rim_slab))
    lid = outer.intersect(lid_side).fuse(rim).clean()
    return tub, lid, powder


def build():
    parts, powder = {}, {}
    tub, lid, pw = tub_and_lid(bottom_region, -ZB + T, lid_above=False)
    parts["bottom_tub"], parts["bottom_lid"], powder["bottom"] = tub, lid, pw
    o, zax = cq.Vector(0, 0, 0), cq.Vector(0, 0, 1)
    for kind in ("x", "y"):
        tub, lid, pw = tub_and_lid(lambda t, o=False, k=kind: block_region(t, k, o), E - T, lid_above=True)
        for name, ang in ((f"pos_{kind}", 0), (f"neg_{kind}", 180)):
            parts[f"block_{name}"] = tub.rotate(o, zax, ang)
            parts[f"block_lid_{name}"] = lid.rotate(o, zax, ang)
            powder[f"block_{name}"] = pw.rotate(o, zax, ang)
    return parts, powder


# --------------------------------------------------------------------- checks
def check(parts, powder, cross):
    lines = []
    add = lines.append
    add("PART VOLUMES (cm^3 of plastic)")
    for k, v in parts.items():
        add(f"  {k:18s} {v.Volume()/1000:7.1f}   solids: {len(v.Solids())}")
    add("POWDER VOLUMES (cm^3)")
    tot = 0.0
    for k, v in powder.items():
        tot += v.Volume() / 1000
        add(f"  {k:18s} {v.Volume()/1000:7.1f}")
    add(f"  total {tot:.0f} cm^3 -> {tot*9/1000:.2f} / {tot*11/1000:.2f} / {tot*12.5/1000:.2f} kg "
        f"at 9 / 11 / 12.5 g/cm^3 packed density")

    add("INTERFERENCE (mm^3; only non-zero pairs are listed)")
    names = list(parts)
    worst = 0.0
    for i, a in enumerate(names):
        v = parts[a].intersect(cross).Volume()
        worst = max(worst, v)
        if v > 1e-6:
            add(f"  {a} x cross: {v:.3f}   <-- OVERLAP")
        for b in names[i + 1:]:
            v = parts[a].intersect(parts[b]).Volume()
            worst = max(worst, v)
            if v > 1e-6:
                add(f"  {a} x {b}: {v:.3f}   <-- OVERLAP")
    add(f"  worst overlap of any part with the cross or another part: {worst:.4f}")

    add("ASSEMBLY MOTION (max overlap along the path, mm^3)")

    def sweep(name, shapes, legs, others):
        worst, pos = 0.0, cq.Vector(0, 0, 0)
        for direction, dist in legs:
            for i in range(1, 31):
                p = pos + cq.Vector(*direction) * (dist * i / 30)
                for s in shapes:
                    moved = s.translate(p)
                    for o in [cross] + others:
                        worst = max(worst, moved.intersect(o).Volume())
            pos = pos + cq.Vector(*direction) * dist
        add(f"  {name}: {worst:.3f}")

    blocks = [v for k, v in parts.items() if k.startswith("block")]
    sweep("bottom tub + lid straight down 150, blocks in place",
          [parts["bottom_tub"], parts["bottom_lid"]], [((0, 0, -1), 150)], blocks)
    others_x = [v for k, v in parts.items() if k.startswith("block") and "pos_x" not in k]
    sweep("+X block out along +x 20, then straight up 150 (tub off, other blocks on)",
          [parts["block_pos_x"], parts["block_lid_pos_x"]], [((1, 0, 0), 20), ((0, 0, 1), 150)], others_x)
    others_y = [v for k, v in parts.items() if k.startswith("block") and "pos_y" not in k]
    sweep("+Y block out along +y 20, then straight up 150 (tub off, other blocks on)",
          [parts["block_pos_y"], parts["block_lid_pos_y"]], [((0, 1, 0), 20), ((0, 0, 1), 150)], others_y)
    return lines


def min_wall(parts, powder):
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    lines = ["MIN PLASTIC BETWEEN POWDER AND THE OUTSIDE OF ITS OWN TUB (mm)"]
    for pw, tub, lid in (("bottom", "bottom_tub", "bottom_lid"),
                         ("block_pos_x", "block_pos_x", "block_lid_pos_x"),
                         ("block_pos_y", "block_pos_y", "block_lid_pos_y")):
        closed = parts[tub].fuse(parts[lid]).fuse(powder[pw]).clean()
        outer = cq.Compound.makeCompound(closed.Faces())
        d = BRepExtrema_DistShapeShape(powder[pw].wrapped, outer.wrapped)
        lines.append(f"  {pw:14s} {d.Value():.2f}")
    return lines


def raytrace(powder, n=40000):
    from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector
    from OCP.gp import gp_Dir, gp_Lin, gp_Pnt

    comp = cq.Compound.makeCompound(list(powder.values()))
    inter = IntCurvesFace_ShapeIntersector()
    inter.Load(comp.wrapped, 1e-7)
    arms = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1)]
    port_cos = math.cos(math.atan(TUBE_R / 70.1))
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
            continue
        path = sum(uniq[k + 1] - uniq[k] for k in range(0, len(uniq), 2))
        cos_arm = max(sum(a * b for a, b in zip(d, arm)) for arm in arms)
        rows.append((d, path, cos_arm, cos_arm > port_cos))
    N = len(rows)
    lines = [f"RAY TRACE from the centre, {N} directions uniform over the sphere"]
    lines.append(f"  {100*sum(1 for r in rows if r[3])/N:5.1f} % leave through a port opening")
    rest = [r for r in rows if not r[3]]
    for thr in (12.7, 10.0, 6.0, 0.01):
        lines.append(f"  {100*sum(1 for r in rest if r[1] >= thr)/N:5.1f} % see >= {thr:5.2f} mm of powder")
    lines.append("  by angle from the nearest arm axis (median / min powder path, mm):")
    for lo, hi in ((20, 30), (30, 35), (35, 40), (40, 45), (45, 55), (55, 90)):
        sel = sorted(r[1] for r in rest if lo <= math.degrees(math.acos(min(1, r[2]))) < hi)
        if sel:
            lines.append(f"    {lo:2d}-{hi:2d} deg: {sel[len(sel)//2]:5.1f} / {sel[0]:5.1f}   "
                         f"({100*len(sel)/N:4.1f} % of sphere)")
    for name, sel in (("upper hemisphere", [r for r in rest if r[0][2] > 0]),
                      ("lower hemisphere", [r for r in rest if r[0][2] <= 0])):
        far = sorted((r for r in sel if r[2] < math.cos(math.radians(40))), key=lambda r: r[1])
        lines.append(f"  {name}, more than 40 deg from every arm ({100*len(far)/N:.1f} % of sphere): "
                     f"median {far[len(far)//2][1]:.1f} mm, "
                     f"{100*sum(1 for r in far if r[1] >= 12.7)/len(far):.0f} % >= 12.7, "
                     f"{100*sum(1 for r in far if r[1] >= 10)/len(far):.0f} % >= 10, "
                     f"{100*sum(1 for r in far if r[1] >= 6)/len(far):.0f} % >= 6")
        lines.append("    thinnest (azimuth, elevation, mm): " + "; ".join(
            f"{math.degrees(math.atan2(r[0][1], r[0][0])):.0f}, {math.degrees(math.asin(r[0][2])):.0f}, {r[1]:.1f}"
            for r in far[:6]))
    down = sorted(r[1] for r in rest if r[0][2] < -0.94)
    lines.append(f"  straight down (within 20 deg of -Z): median {down[len(down)//2]:.1f}, min {down[0]:.1f} mm")
    return lines


# ------------------------------------------------------------------- drawings
def wire_points(wire, proj):
    from OCP.BRepAdaptor import BRepAdaptor_CompCurve
    from OCP.GCPnts import GCPnts_QuasiUniformDeflection
    disc = GCPnts_QuasiUniformDeflection(BRepAdaptor_CompCurve(wire.wrapped), 0.05)
    return [proj(cq.Vector(disc.Value(i))) for i in range(1, disc.NbPoints() + 1)]


def sections(parts, powder, cross):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch, Polygon
    from matplotlib.path import Path

    t = 0.02
    diag = box(-BIG, BIG, -t, t, -BIG, BIG).rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), 45)
    views = [
        ("plan at z = 0 (through the arms)", box(-BIG, BIG, -BIG, BIG, -t, t),
         lambda p: (p.x, p.y), lambda n: n.z, ("x", "y")),
        ("plan at z = +32 (blocks around the Z tube)", box(-BIG, BIG, -BIG, BIG, 32 - t, 32 + t),
         lambda p: (p.x, p.y), lambda n: n.z, ("x", "y")),
        ("elevation at y = 0 (through X and Z arms)", box(-BIG, BIG, -t, t, -BIG, BIG),
         lambda p: (p.x, p.z), lambda n: n.y, ("x", "z")),
        ("elevation at x = 33 (across the +X arm)", box(33 - t, 33 + t, -BIG, BIG, -BIG, BIG),
         lambda p: (p.y, p.z), lambda n: n.x, ("y", "z")),
        ("diagonal elevation x = y", diag,
         lambda p: ((p.x + p.y) / math.sqrt(2), p.z), lambda n: (n.y - n.x) / math.sqrt(2),
         ("along x = y", "z")),
    ]
    groups = [("cross (steel)", [cross], "#8a8f98"),
              ("tungsten powder", list(powder.values()), "#2b3a55"),
              ("bottom tub + lid", [parts["bottom_tub"], parts["bottom_lid"]], "#d9822b"),
              ("X blocks + lids", [v for k, v in parts.items() if k.endswith("_x")], "#3a9a6b"),
              ("Y blocks + lids", [v for k, v in parts.items() if k.endswith("_y")], "#b0509a")]
    fig, axes = plt.subplots(2, 3, figsize=(19, 12.5))
    axes = axes.ravel()
    for ax, (title, cutter, proj, ncomp, labels) in zip(axes, views):
        for _, shapes, color in groups:
            for shp in shapes:
                for f in shp.intersect(cutter).Faces():
                    if ncomp(f.normalAt()) < 0.9:
                        continue
                    verts, codes = [], []
                    for w in [f.outerWire()] + list(f.innerWires()):
                        pts = wire_points(w, proj)
                        if len(pts) >= 3:
                            verts += pts + [pts[0]]
                            codes += [Path.MOVETO] + [Path.LINETO] * (len(pts) - 1) + [Path.CLOSEPOLY]
                    if verts:
                        ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="k", lw=0.4))
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(labels[0] + " (mm)"); ax.set_ylabel(labels[1] + " (mm)")
        ax.set_aspect("equal"); ax.set_xlim(-85, 85); ax.set_ylim(-60, 80)
        ax.grid(True, lw=0.3, alpha=0.5)
    ax = axes[-1]
    ax.axis("off")
    for i, (label, _, color) in enumerate(groups):
        y = 0.85 - 0.12 * i
        ax.add_patch(Polygon([(0.05, y), (0.15, y), (0.15, y + 0.07), (0.05, y + 0.07)], color=color))
        ax.text(0.2, y + 0.02, label, fontsize=14)
    fig.tight_layout()
    path = os.path.join(OUT, "sections.png")
    fig.savefig(path, dpi=110)
    return path


def main():
    cross = cq.importers.importStep(CROSS_STEP).val()
    parts, powder = build()
    printable = {"bottom_tub": "bottom_tub", "bottom_lid": "bottom_lid",
                 "block_pos_x": "x_block_x2", "block_lid_pos_x": "x_block_lid_x2",
                 "block_pos_y": "y_block_x2", "block_lid_pos_y": "y_block_lid_x2"}
    for key, name in printable.items():
        cq.exporters.export(cq.Workplane(obj=parts[key]), os.path.join(OUT, f"{name}.step"))
        cq.exporters.export(cq.Workplane(obj=parts[key]), os.path.join(OUT, f"{name}.stl"),
                            tolerance=0.02, angularTolerance=0.1)
    asm = cq.Assembly()
    asm.add(cross, name="cross", color=cq.Color(0.6, 0.6, 0.65))
    for name, shp in parts.items():
        col = (cq.Color(0.85, 0.5, 0.17) if name.startswith("bottom")
               else cq.Color(0.23, 0.6, 0.42) if name.endswith("_x") else cq.Color(0.7, 0.3, 0.6))
        asm.add(shp, name=name, color=col)
    for name, shp in powder.items():
        asm.add(shp, name="powder_" + name, color=cq.Color(0.17, 0.23, 0.33))
    asm.export(os.path.join(OUT, "shield_assembly.step"))

    report = check(parts, powder, cross) + min_wall(parts, powder) + raytrace(powder)
    text = "\n".join(report)
    print(text)
    with open(os.path.join(OUT, "report.txt"), "w") as fh:
        fh.write(text + "\n")
    if "--no-sections" not in sys.argv:
        print("sections ->", sections(parts, powder, cross))


if __name__ == "__main__":
    main()
