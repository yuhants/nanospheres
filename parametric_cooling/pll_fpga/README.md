# PLL FPGA design (vendored)

This directory is a **vendored copy** of the Red Pitaya dual-PLL FPGA design used
for parametric feedback cooling. It is kept in-tree so that the precompiled
bitstream can be uploaded to the board by `../pll_control.py`, and so the FPGA
can be *modified and recompiled* later without re-cloning anything.

## Provenance

Source: https://github.com/tefelixhu/redpitayapll (the original design), which is
also the basis of the C++ fork https://github.com/vhock/Phase-locked-Loop.

Developed at the ETH Zürich Photonics Lab for parametric cooling of a levitated
nanoparticle. Original authors: Felix Tebbenjohanns (hardware/VHDL),
Dominik Windey & Markus Rademacher (server program). Licensed under **GPLv3**
(see `LICENSE`). Our Python driver/GUI (`../pll_control.py`, `../pll_gui.py`) is
an independent reimplementation of the host-side software against this design's
register map.

## What's here

```
bitfile/pll_project.bit   Precompiled bitstream — this is what gets loaded onto the board.
vhdl/                     VHDL sources for the PLL building blocks.
cores/                    Custom IP cores (axi_* / axis_* wrappers).
cfg/                      Pin / clock constraints and board XML.
scripts/                  Block-design + core build Tcl.
create_project.tcl        Regenerate the Vivado project (use Vivado 2017.2).
make_cores.tcl            Package the custom cores.
doc/regs.{pdf,xlsx}       Authoritative register map.
doc/figures/              Top-level + PLL schematics.
```

## Using the bitstream (no FPGA tools needed)

`pll_control.py` SCPs `bitfile/pll_project.bit` to `/tmp/` on the board and runs
`cat /tmp/pll_project.bit > /dev/xdevcfg` to load it. Note this **replaces** the
stock Red Pitaya FPGA image (and any PyRPL bitstream), so the SCPI server /
`pyrpl_control.py` cannot be used at the same time.

## Rebuilding the bitstream (Option 2 — modifying the FPGA)

You only need this if you want to change the FPGA logic itself.

1. Install **Vivado 2017.2** (the design is version-sensitive).
2. From this directory, in the Vivado Tcl console: `source ./create_project.tcl`
   (creates `scripts/tmp/pll_project`).
3. Edit the block design / VHDL, then generate a bitstream.
4. Copy the new `*.bit` over `bitfile/pll_project.bit`.

If you change the **register map**, update both `doc/regs.*` and the
`PARAM_MAP` / base addresses in `../pll_control.py` to match.
