"""
Which sections the lab GUI shows, where they sit, and whether they start folded.

To add a task: write ``lab_gui/panels/<something>_panel.py`` with a class that
subclasses ``TaskPanel``, then add one line to ``PANEL_SPECS``.  Nothing else in
the GUI needs to know about it — the window drives every panel through the
TaskPanel methods.

Everything is on one page, so a spec also says which column a section goes in.
Layout is a per-machine matter (screen size, which instruments this PC even
has), so ``lab_gui/layout.json`` — if it exists — overrides the defaults below
without editing code:

    {
      "columns": 2,
      "sections": {
        "EFieldPanel":     {"column": 0, "enabled": true,  "collapsed": false},
        "HVPanel":         {"column": 0},
        "PulseTaskPanel":  {"enabled": false}
      }
    }

The file is optional and partial: any key you leave out keeps its default here.
"Sections" in the menu bar toggles visibility live, and "Save layout" writes the
current arrangement back to that file.

Panels are imported lazily and one at a time, so a section whose driver is
missing (no pyvisa, no pyserial, instrument library not installed on this
machine) becomes a section explaining why, instead of stopping the whole GUI
from starting.
"""

import importlib
import json
import os
import traceback
from dataclasses import dataclass
from typing import Dict, List, Tuple

from PyQt5.QtWidgets import QVBoxLayout, QLabel, QTextEdit

from lab_gui.task_panel import TaskPanel


LAYOUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'layout.json')

DEFAULT_COLUMNS = 2


@dataclass
class PanelSpec:
    module:    str
    cls:       str
    column:    int  = 0
    enabled:   bool = True
    collapsed: bool = False

    @property
    def key(self) -> str:
        """Name used in layout.json and in presets."""
        return self.cls


PANEL_SPECS: List[PanelSpec] = [
    PanelSpec('lab_gui.panels.efield_panel',     'EFieldPanel',    column=0),
    PanelSpec('lab_gui.panels.hv_panel',         'HVPanel',        column=0),
    PanelSpec('lab_gui.panels.pulse_task_panel', 'PulseTaskPanel', column=1),
]


class BrokenPanel(TaskPanel):
    """Placeholder for a panel that would not import."""

    def __init__(self, title: str, error: str, parent=None):
        super().__init__(parent)
        self.title = f'{title} (unavailable)'
        layout = QVBoxLayout(self)
        label  = QLabel(f'{title} could not be loaded.')
        label.setStyleSheet('font-weight: bold; color: #b00;')
        detail = QTextEdit(error)
        detail.setReadOnly(True)
        detail.setMaximumHeight(160)
        layout.addWidget(label)
        layout.addWidget(detail)


def load_layout(path: str = LAYOUT_PATH) -> Tuple[List[PanelSpec], int]:
    """
    The specs to build, with ``layout.json`` applied on top, and the column count.

    A malformed or unreadable file is ignored rather than fatal — the GUI is more
    useful with a default layout than not at all.
    """
    specs   = [PanelSpec(**vars(s)) for s in PANEL_SPECS]      # copies
    columns = DEFAULT_COLUMNS
    try:
        with open(path) as f:
            config = json.load(f)
    except FileNotFoundError:
        return specs, columns
    except Exception:
        return specs, columns

    columns = int(config.get('columns', columns))
    sections: Dict[str, dict] = config.get('sections', {})
    for spec in specs:
        section = sections.get(spec.key, {})
        spec.column    = int(section.get('column', spec.column))
        spec.enabled   = bool(section.get('enabled', spec.enabled))
        spec.collapsed = bool(section.get('collapsed', spec.collapsed))
    columns = max(1, max([columns] + [s.column + 1 for s in specs]))
    return specs, columns


def save_layout(specs: List[PanelSpec], columns: int, path: str = LAYOUT_PATH) -> None:
    """Write the current arrangement back to layout.json."""
    config = {
        'columns': columns,
        'sections': {s.key: {'column': s.column, 'enabled': s.enabled,
                             'collapsed': s.collapsed} for s in specs},
    }
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)


def load_panels(specs: List[PanelSpec]) -> List[Tuple[PanelSpec, TaskPanel]]:
    """
    Instantiate every spec, in order.

    Disabled sections are built too but left hidden, so the Sections menu can
    bring one back without a restart.  Building a panel opens no ports — that
    only happens when you press Connect — so an unused section costs nothing but
    the import.
    """
    built = []
    for spec in specs:
        try:
            module = importlib.import_module(spec.module)
            panel  = getattr(module, spec.cls)()
        except Exception:
            panel = BrokenPanel(spec.cls, traceback.format_exc())
        built.append((spec, panel))
    return built
