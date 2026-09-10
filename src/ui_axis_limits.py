"""Synchronized numeric bounds and zoom controls for one plot axis."""

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QDoubleSpinBox, QSlider, QPushButton
from PySide6.QtCore import Qt


class AxisLimitsControl(QWidget):
    """Edit one axis without changing another axis or the camera."""

    limitsChanged = Signal(float, float)

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.home = (0.0, 1.0)
        self.current = self.home
        self._syncing = False
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(title)
        self.label.setMinimumWidth(115)
        row.addWidget(self.label)
        self.lower = self._bound_field("Minimum")
        self.upper = self._bound_field("Maximum")
        for name, control in (("Min", self.lower), ("Max", self.upper)):
            row.addWidget(QLabel(name))
            row.addWidget(control)
        row.addWidget(QLabel("Zoom"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(-1000, 3000)
        self.slider.setToolTip("Logarithmic axis zoom, centered on the current range")
        row.addWidget(self.slider, 1)
        self.zoom = QDoubleSpinBox()
        self.zoom.setRange(0.1, 1000.0)
        self.zoom.setDecimals(3)
        self.zoom.setSingleStep(0.1)
        self.zoom.setSuffix("x")
        self.zoom.setKeyboardTracking(False)
        row.addWidget(self.zoom)
        self.reset_button = QPushButton("Reset")
        row.addWidget(self.reset_button)
        self.lower.editingFinished.connect(self._bounds_edited)
        self.upper.editingFinished.connect(self._bounds_edited)
        self.zoom.valueChanged.connect(self._zoom_changed)
        self.slider.valueChanged.connect(lambda value: self._zoom_changed(10 ** (value / 1000)))
        self.reset_button.clicked.connect(lambda: self.limitsChanged.emit(*self.home))
        self.set_limits(self.home)

    def _bound_field(self, name):
        control = QDoubleSpinBox()
        control.setRange(-1e100, 1e100)
        control.setDecimals(9)
        control.setKeyboardTracking(False)
        control.setToolTip(name + " axis limit; minimum must be below maximum")
        control.setMinimumWidth(105)
        return control

    def set_limits(self, limits, home=None):
        self._syncing = True
        try:
            if home is not None:
                self.home = tuple(home)
            self.current = tuple(limits)
            self.lower.setValue(limits[0])
            self.upper.setValue(limits[1])
            step = max(abs(limits[1] - limits[0]) / 100, 1e-9)
            self.lower.setSingleStep(step)
            self.upper.setSingleStep(step)
            factor = (self.home[1] - self.home[0]) / (limits[1] - limits[0])
            self.zoom.setValue(factor)
            self.slider.setValue(round(np.log10(max(factor, 1e-12)) * 1000))
            self.zoom.setToolTip(f"Actual zoom: {factor:.6g}x; slider range is 0.1–1000x")
        finally:
            self._syncing = False

    def _bounds_edited(self):
        if self._syncing:
            return
        lower, upper = self.lower.value(), self.upper.value()
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            self.set_limits(self.current)
            self.lower.setToolTip("Invalid range restored: minimum must be below maximum")
            return
        if (lower, upper) != self.current:
            self.limitsChanged.emit(lower, upper)

    def _zoom_changed(self, factor):
        if self._syncing:
            return
        span = (self.home[1] - self.home[0]) / factor
        center = (self.current[0] + self.current[1]) / 2
        self.limitsChanged.emit(center - span / 2, center + span / 2)
