from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QSlider, QSpinBox, QDoubleSpinBox
)
from PySide6.QtCore import Qt, Signal

class SliderSpinBox(QWidget):
    """
    A synchronized Slider + SpinBox widget for numeric parameter control.
    Supports both int and float values.
    """
    valueChanged = Signal(float)

    def __init__(self, label_text, min_val, max_val, initial_val, step=1, is_float=False, decimals=2):
        super().__init__()
        self.is_float = is_float
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Label
        self.label = QLabel(label_text)
        self.label.setMinimumWidth(80)
        layout.addWidget(self.label)
        
        # Slider
        self.slider = QSlider(Qt.Horizontal)
        # Sliders are int only, so we scale if float
        self.scale_factor = 10**decimals if is_float else 1
        
        self.slider.setRange(int(min_val * self.scale_factor), int(max_val * self.scale_factor))
        self.slider.setValue(int(initial_val * self.scale_factor))
        layout.addWidget(self.slider)
        
        # SpinBox
        if is_float:
            self.spinbox = QDoubleSpinBox()
            self.spinbox.setDecimals(decimals)
            self.spinbox.setSingleStep(step)
        else:
            self.spinbox = QSpinBox()
            self.spinbox.setSingleStep(int(step))
            
        self.spinbox.setRange(min_val, max_val)
        self.spinbox.setValue(initial_val)
        self.spinbox.setFixedWidth(70)
        layout.addWidget(self.spinbox)
        
        # Connect signals
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.spinbox.valueChanged.connect(self._on_spinbox_changed)

    def _on_slider_changed(self, val):
        real_val = val / self.scale_factor
        real_val = round(real_val, 2) if self.is_float else real_val
        self.spinbox.blockSignals(True)
        self.spinbox.setValue(real_val)
        self.spinbox.blockSignals(False)
        self.valueChanged.emit(real_val)

    def _on_spinbox_changed(self, val):
        slider_val = int(val * self.scale_factor)
        self.slider.blockSignals(True)
        self.slider.setValue(slider_val)
        self.slider.blockSignals(False)
        self.valueChanged.emit(val)

    def value(self):
        return self.spinbox.value()
        
    def set_range(self, min_val, max_val):
        """Update range for both slider and spinbox dynamically."""
        self.spinbox.blockSignals(True)
        self.slider.blockSignals(True)
        
        self.spinbox.setRange(min_val, max_val)
        
        # Safe cast for QSlider (32-bit signed int limit)
        # 2147483647 is INT_MAX
        slider_min = int(min_val * self.scale_factor)
        slider_max = int(max_val * self.scale_factor)
        
        MAX_INT = 2147483647
        MIN_INT = -2147483648
        
        # Clamp to avoid overflow crash
        if slider_max > MAX_INT:
            slider_max = MAX_INT
        if slider_min < MIN_INT:
            slider_min = MIN_INT
            
        # Ensure min <= max after clamping
        if slider_min > slider_max:
            slider_min = slider_max
            
        self.slider.setRange(slider_min, slider_max)
        
        self.spinbox.blockSignals(False)
        self.slider.blockSignals(False)
