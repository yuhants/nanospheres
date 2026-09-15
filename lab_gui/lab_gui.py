"""
Lab task GUI — the routine "switch this on for a bit" jobs in one window.

Everything is on a single page: each task is a section you can fold away, not a
tab, so the E field, the HV trigger and the pulses are all in front of you at
once.  In practice these get used together — strike the needle, watch the charge
step, drive the sphere again — and that is awkward when each one is hidden
behind a tab.

The sections replace scripts you would otherwise edit at the top and run:

  E field            control/charge_measurement.py   (Tektronix AFG1022)
  High voltage       control/apply_hv.py             (RIGOL DG822)
  Calibration pulses pulse_gen/pulse_panel.py        (Quantum Composers 9614)

What the window itself provides, on top of the sections:

  * one log for everything, so an HV strike and an E-field change appear in the
    same timeline;
  * a status bar naming every output that is on, and a red "●" on the section
    heading of each — including folded ones;
  * "All outputs OFF" (also the Escape key), which calls ``stop()`` on every
    panel, connected or not;
  * presets: one JSON holding every section's settings.

Layout is configurable: the Sections menu shows, hides and folds sections and
moves them between columns, and "Save layout" persists that to
``lab_gui/layout.json`` for next time.  See lab_gui/registry.py for the file
format and for how to add a task.

Instruments are opened per section, only when you press Connect, and closed on
exit.  Nothing here writes data files — run the DAQ separately.

Run:
    python lab_gui/lab_gui.py
"""

import json
import os
import sys
import time
from typing import List, Tuple

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
    QTextEdit, QPushButton, QLabel, QGroupBox, QFileDialog, QMessageBox,
    QShortcut, QAction, QFrame, QSplitter,
)
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import Qt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lab_gui.registry import load_layout, load_panels, save_layout, PanelSpec
from lab_gui.task_panel import TaskPanel, SectionFrame, PRESETS_DIR


class LabMainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Lab tasks')
        self.setMinimumSize(1000, 800)

        self._specs, self._columns = load_layout()
        self._built: List[Tuple[PanelSpec, TaskPanel]] = load_panels(self._specs)
        self._sections: List[SectionFrame] = []

        self._build_ui()
        self._refresh_status()

    @property
    def _panels(self) -> List[TaskPanel]:
        return [panel for _, panel in self._built]

    # ── UI construction ─────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        layout  = QVBoxLayout(central)

        layout.addWidget(self._build_page(), stretch=1)
        layout.addWidget(self._build_log_group())
        layout.addWidget(self._build_control_bar())
        self.setCentralWidget(central)

        self._build_menu()

        # Escape is the panic key, from anywhere in the window.
        QShortcut(QKeySequence(Qt.Key_Escape), self, self._all_off)

    def _build_page(self) -> QScrollArea:
        """One scrolling page of sections, laid out in columns."""
        page = QWidget()
        page_layout = QHBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)

        # A splitter rather than a fixed grid, so column widths can be dragged —
        # the pulse section is much wider than the others.
        splitter = QSplitter(Qt.Horizontal)
        column_layouts = []
        for _ in range(self._columns):
            holder = QWidget()
            column = QVBoxLayout(holder)
            column.setContentsMargins(0, 0, 0, 0)
            column_layouts.append(column)
            splitter.addWidget(holder)
        page_layout.addWidget(splitter)

        for spec, panel in self._built:
            section = SectionFrame(panel, collapsed=spec.collapsed)
            section.setVisible(spec.enabled)
            self._sections.append(section)
            column = column_layouts[min(spec.column, self._columns - 1)]
            column.addWidget(section)

            panel.log_message.connect(
                lambda text, p=panel: self._log_line(p.title, text))
            panel.state_changed.connect(self._refresh_status)

        for column in column_layouts:
            column.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(page)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        return scroll

    def _build_menu(self) -> None:
        menu = self.menuBar().addMenu('&Sections')
        self._section_actions = []
        for spec, panel in self._built:
            action = QAction(panel.title, self, checkable=True)
            action.setChecked(spec.enabled)
            action.toggled.connect(
                lambda shown, s=spec: self._set_section_visible(s, shown))
            menu.addAction(action)
            self._section_actions.append(action)
        menu.addSeparator()
        for label, collapsed in [('Fold all', True), ('Unfold all', False)]:
            action = QAction(label, self)
            action.triggered.connect(lambda _, c=collapsed: self._fold_all(c))
            menu.addAction(action)
        menu.addSeparator()
        save_action = QAction('Save layout', self)
        save_action.setToolTip('Remember the current sections and folds in '
                               'lab_gui/layout.json')
        save_action.triggered.connect(self._save_layout)
        menu.addAction(save_action)

    def _build_log_group(self) -> QGroupBox:
        group  = QGroupBox('Log')
        layout = QVBoxLayout(group)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(130)
        layout.addWidget(self._log)
        return group

    def _build_control_bar(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)

        self._all_off_btn = QPushButton('■  All outputs OFF  (Esc)')
        self._all_off_btn.setStyleSheet('font-weight: bold;')
        self._all_off_btn.clicked.connect(self._all_off)

        self._status_label = QLabel('')
        self._status_label.setWordWrap(True)

        layout.addWidget(self._all_off_btn)
        layout.addWidget(self._status_label, stretch=1)
        for label, slot in [('Save preset', self._save_preset),
                            ('Load preset', self._load_preset)]:
            button = QPushButton(label)
            button.clicked.connect(slot)
            layout.addWidget(button)
        return widget

    # ── Layout ──────────────────────────────────────────────────────────────────

    def _section_for(self, spec: PanelSpec) -> SectionFrame:
        for (candidate, _), section in zip(self._built, self._sections):
            if candidate is spec:
                return section
        raise KeyError(spec.key)

    def _set_section_visible(self, spec: PanelSpec, shown: bool) -> None:
        section = self._section_for(spec)
        if not shown and section.panel.is_active():
            # Hiding a section does not stop it, so refuse rather than let an
            # output disappear off the page.
            QMessageBox.warning(self, 'Section in use',
                                f'{section.panel.title} is still driving — '
                                'turn its output off before hiding it.')
            self._sync_menu()
            return
        spec.enabled = shown
        section.setVisible(shown)

    def _sync_menu(self) -> None:
        """Put the tick boxes back in step after a refused hide."""
        for action, (spec, _) in zip(self._section_actions, self._built):
            action.blockSignals(True)
            action.setChecked(spec.enabled)
            action.blockSignals(False)

    def _fold_all(self, collapsed: bool) -> None:
        for section in self._sections:
            section.set_collapsed(collapsed)

    def _save_layout(self) -> None:
        for spec, section in zip([s for s, _ in self._built], self._sections):
            spec.collapsed = section.is_collapsed
        try:
            save_layout([s for s, _ in self._built], self._columns)
        except Exception as exc:
            QMessageBox.warning(self, 'Save layout', str(exc))
            return
        self._log_line('layout', 'saved to lab_gui/layout.json')

    # ── State ───────────────────────────────────────────────────────────────────

    def _active_panels(self) -> List[TaskPanel]:
        return [p for p in self._panels if p.is_active()]

    def _refresh_status(self) -> None:
        for section in self._sections:
            section.refresh_status()

        active = self._active_panels()
        if active:
            lines = [p.status_text() or f'{p.title}: on' for p in active]
            self._status_label.setText(' · '.join(lines))
            self._status_label.setStyleSheet('color: #b00; font-weight: bold;')
        else:
            self._status_label.setText('all outputs off')
            self._status_label.setStyleSheet('color: gray;')

    def _log_line(self, source: str, text: str) -> None:
        self._log.append(f'[{time.strftime("%H:%M:%S")}] {source}: {text}')

    def _all_off(self) -> None:
        for panel in self._panels:
            try:
                panel.stop()
            except Exception as exc:                     # a panel must not block
                self._log_line(panel.title, f'stop failed: {exc}')  # the others
        self._refresh_status()

    # ── Presets ─────────────────────────────────────────────────────────────────

    def _preset_key(self, panel: TaskPanel) -> str:
        return type(panel).__name__

    def _save_preset(self) -> None:
        os.makedirs(PRESETS_DIR, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save preset', PRESETS_DIR, 'JSON (*.json)')
        if not path:
            return
        data = {self._preset_key(p): p.to_dict() for p in self._panels}
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        self._log_line('presets', f'saved to {path}')

    def _load_preset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load preset', PRESETS_DIR, 'JSON (*.json)')
        if not path:
            return
        with open(path) as f:
            data = json.load(f)
        for panel in self._panels:
            section = data.get(self._preset_key(panel))
            if section:
                try:
                    panel.from_dict(section)
                except Exception as exc:
                    self._log_line(panel.title, f'preset not applied: {exc}')
        self._log_line('presets', f'loaded from {path}')

    # ── Shutdown ────────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        active = self._active_panels()
        if active:
            names = ', '.join(p.title for p in active)
            answer = QMessageBox.question(
                self, 'Outputs still on',
                f'These are still driving: {names}.\n\nTurn everything off and quit?',
                QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Yes)
            if answer != QMessageBox.Yes:
                event.ignore()
                return
        for panel in self._panels:
            try:
                panel.shutdown()
            except Exception:
                pass
        super().closeEvent(event)


# ── Entry points ───────────────────────────────────────────────────────────────

def launch_gui() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    window = LabMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    launch_gui()
