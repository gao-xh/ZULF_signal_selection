"""Independent time-frequency surface with constant-time spectrum slices."""

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QPushButton, QSplitter, QScrollArea, QComboBox,
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from src.ui_components import SliderSpinBox
from src.ui_axis_limits import AxisLimitsControl
from src.config import (
    STFT_3D_TIME_LINES, STFT_3D_MAX_TIME_LINES, STFT_3D_SURFACE_ALPHA,
    STFT_3D_MAX_TIME_POINTS, STFT_3D_MAX_FREQUENCY_POINTS,
)


class StftSurfaceWindow(QMainWindow):
    """Display cached STFT magnitudes without recomputing or normalizing slices."""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("STFT 3D Surface — Time / Frequency / Amplitude")
        self.resize(1120, 780)
        self._data = None
        self._camera = (28, -58)
        self._view_limits = {}
        self._home_limits = {}
        self._reset_limits = True
        self._last_log_scale = False
        self.slice_indices = np.array([], dtype=int)
        self.slice_artists = []
        self.surface = None
        root = QSplitter(Qt.Vertical)
        self.setCentralWidget(root)
        controls = QWidget()
        form = QVBoxLayout(controls)
        self.caption = QLabel("No spectrogram data. Use Show 3D Surface in the main window.")
        self.caption.setWordWrap(True)
        form.addWidget(self.caption)
        self.line_count = SliderSpinBox("Time slices", 1, STFT_3D_MAX_TIME_LINES,
                                       STFT_3D_TIME_LINES, step=1)
        self.opacity = SliderSpinBox("Surface opacity", 0.1, 1.0,
                                    STFT_3D_SURFACE_ALPHA, step=0.05,
                                    is_float=True, decimals=2)
        form.addWidget(self.line_count)
        form.addWidget(self.opacity)
        row = QHBoxLayout()
        self.show_lines = QCheckBox("Show time-slice spectra")
        self.show_lines.setChecked(True)
        self.log_scale = QCheckBox("Amplitude in dB")
        self.reset_button = QPushButton("Reset 3D view")
        row.addWidget(self.show_lines)
        row.addWidget(self.log_scale)
        row.addStretch()
        row.addWidget(self.reset_button)
        form.addLayout(row)
        self.axis_controls = {}
        for name, title in (("x", "X: Time (s)"), ("y", "Y: Frequency (Hz)"), ("z", "Z: Amplitude")):
            control = AxisLimitsControl(title)
            control.limitsChanged.connect(lambda lower, upper, axis=name: self.set_axis_limits(axis, lower, upper))
            self.axis_controls[name] = control
            form.addWidget(control)
        navigation = QHBoxLayout()
        navigation.addWidget(QLabel("Mouse wheel zoom"))
        self.wheel_axis = QComboBox()
        self.wheel_axis.addItems(["X: Time", "Y: Frequency", "Z: Amplitude"])
        navigation.addWidget(self.wheel_axis)
        navigation.addStretch()
        reset_axes = QPushButton("Reset all axes")
        reset_axes.clicked.connect(self.reset_axis_limits)
        navigation.addWidget(reset_axes)
        form.addLayout(navigation)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        root.addWidget(scroll)
        graph = QWidget()
        layout = QVBoxLayout(graph)
        layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.mpl_connect("scroll_event", self._scroll_axis)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        root.addWidget(graph)
        root.setStretchFactor(0, 0)
        root.setStretchFactor(1, 1)
        root.setSizes([285, 495])
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(160)
        self.timer.timeout.connect(self.redraw)
        for control in (self.line_count, self.opacity):
            control.valueChanged.connect(lambda _: self.timer.start())
        for control in (self.show_lines, self.log_scale):
            control.toggled.connect(lambda _: self.timer.start())
        self.reset_button.clicked.connect(self.reset_view)
        self.redraw()

    def set_data(self, times, frequencies, magnitude, frequency_range, folded=False):
        """Snapshot the displayed frequency range; preserve acquisition time offsets."""
        times = np.asarray(times, dtype=float)
        frequencies = np.asarray(frequencies, dtype=float)
        magnitude = np.asarray(magnitude, dtype=float)
        if times.ndim != 1 or frequencies.ndim != 1 or magnitude.shape != (len(frequencies), len(times)):
            raise ValueError("STFT matrix must have shape (frequency, time)")
        if not np.isfinite(times).all() or not np.isfinite(frequencies).all() or not np.isfinite(magnitude).all():
            raise ValueError("STFT data must contain finite values")
        if np.any(magnitude < 0) or np.any(np.diff(times) <= 0) or np.any(np.diff(frequencies) <= 0):
            raise ValueError("STFT magnitudes must be nonnegative and coordinates increasing")
        lower, upper = frequency_range
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            raise ValueError("Choose an increasing, finite frequency range")
        mask = (frequencies >= lower) & (frequencies <= upper)
        if len(times) < 2 or mask.sum() < 2:
            raise ValueError("3D surface needs at least two time frames and two visible frequency bins")
        self._data = (times.copy(), frequencies[mask].copy(), magnitude[mask].copy(), folded)
        self._reset_limits = True
        self.redraw()

    def invalidate(self):
        """Clear outdated data after the processed source changes."""
        self.timer.stop()
        self._data = None
        self._reset_limits = True
        self.redraw()
        self.caption.setText("Source changed. Click Show 3D Surface to refresh the spectrogram.")

    def reset_view(self):
        self._camera = (28, -58)
        if hasattr(self, "axis"):
            self.axis.view_init(elev=self._camera[0], azim=self._camera[1])
        self.redraw()

    def set_axis_limits(self, name, lower, upper):
        """Zoom or pan one axis, leaving all other limits and the camera intact."""
        if self._data is None or not np.isfinite([lower, upper]).all() or lower >= upper:
            return
        self.toolbar.push_current()
        getattr(self.axis, f"set_{name}lim")(lower, upper)
        self.toolbar.push_current()
        self.canvas.draw_idle()

    def _axis_limits_changed(self, name):
        limits = getattr(self.axis, f"get_{name}lim")()
        self._view_limits[name] = limits
        self.axis_controls[name].set_limits(limits, self._home_limits[name])

    def reset_axis_limits(self):
        if self._data is None:
            return
        self.toolbar.push_current()
        for name, limits in self._home_limits.items():
            getattr(self.axis, f"set_{name}lim")(*limits)
        self.toolbar.push_current()
        self.canvas.draw_idle()

    def _scroll_axis(self, event):
        if self._data is None or event.inaxes is not self.axis or not np.isfinite(event.step):
            return
        name = "xyz"[self.wheel_axis.currentIndex()]
        lower, upper = getattr(self.axis, f"get_{name}lim")()
        half_span = (upper - lower) / (2 * 1.2 ** np.clip(event.step, -10, 10))
        center = (lower + upper) / 2
        self.set_axis_limits(name, center - half_span, center + half_span)

    def redraw(self):
        if hasattr(self, "axis"):
            self._camera = (self.axis.elev, self.axis.azim)
            if not self._reset_limits:
                self._view_limits = {name: getattr(self.axis, f"get_{name}lim")() for name in "xyz"}
        if self._reset_limits:
            self._view_limits.clear()
            self._home_limits.clear()
        elif self.log_scale.isChecked() != self._last_log_scale:
            # Linear amplitude and dB need different Z bounds; keep X and Y.
            self._view_limits.pop("z", None)
            self._home_limits.pop("z", None)
        self._last_log_scale = self.log_scale.isChecked()
        self.figure.clear()
        self.axis = self.figure.add_subplot(111, projection="3d")
        self.axis.view_init(elev=self._camera[0], azim=self._camera[1])
        self.axis.set_xlabel("Time (s)", labelpad=10)
        self.axis.set_ylabel("Frequency (Hz)", labelpad=10)
        self.axis.set_zlabel("Amplitude", labelpad=10)
        self.slice_indices = np.array([], dtype=int)
        self.slice_artists = []
        self.surface = None
        if self._data is None:
            for control in self.axis_controls.values():
                control.setEnabled(False)
            self.axis.set_title("STFT 3D Surface — no current data")
            self.canvas.draw_idle()
            return
        times, frequencies, magnitude, folded = self._data
        count = min(int(self.line_count.value()), len(times))
        self.slice_indices = np.unique(np.linspace(0, len(times) - 1, count).round().astype(int))
        # Rendering-only sampling. Slice times are included in the surface grid.
        time_indices = np.unique(np.concatenate((
            np.linspace(0, len(times) - 1, min(len(times), STFT_3D_MAX_TIME_POINTS)).round().astype(int),
            self.slice_indices,
        )))
        freq_indices = np.unique(np.linspace(0, len(frequencies) - 1,
                                            min(len(frequencies), STFT_3D_MAX_FREQUENCY_POINTS)).round().astype(int))
        displayed = 20 * np.log10(magnitude + 1e-12) if self.log_scale.isChecked() else magnitude
        unit = "Amplitude (dB re 1 input unit)" if self.log_scale.isChecked() else "Amplitude (input units)"
        x, y = np.meshgrid(times[time_indices], frequencies[freq_indices])
        z = displayed[np.ix_(freq_indices, time_indices)]
        self.surface = self.axis.plot_surface(x, y, z, cmap="viridis", linewidth=0,
                                              antialiased=True, alpha=float(self.opacity.value()),
                                              rstride=1, cstride=1)
        if self.show_lines.isChecked():
            for index in self.slice_indices:
                # Full frequency resolution for each selected spectrum, no offsets.
                line, = self.axis.plot(np.full(len(frequencies), times[index]), frequencies,
                                       displayed[:, index], color="#142634", linewidth=0.85,
                                       alpha=0.95)
                self.slice_artists.append(line)
        self.axis.set_ylabel("Absolute frequency (Hz)" if folded else "Frequency (Hz)", labelpad=10)
        self.axis.set_zlabel(unit, labelpad=10)
        self.axis.set_title("STFT surface with constant-time spectrum slices")
        self.figure.colorbar(self.surface, ax=self.axis, shrink=0.65, pad=0.1, label=unit)
        self.axis_controls["z"].label.setText("Z: Amplitude (dB)" if self.log_scale.isChecked() else "Z: Amplitude")
        for name, control in self.axis_controls.items():
            control.setEnabled(True)
            self._home_limits.setdefault(name, getattr(self.axis, f"get_{name}lim")())
            if name in self._view_limits:
                getattr(self.axis, f"set_{name}lim")(*self._view_limits[name])
            self._axis_limits_changed(name)
            self.axis.callbacks.connect(f"{name}lim_changed", lambda axis, key=name: self._axis_limits_changed(key))
        self._reset_limits = False
        sampled = len(freq_indices) < len(frequencies) or len(time_indices) < len(times)
        self.caption.setText(
            f"{len(times)} time frames × {len(frequencies)} frequency bins | "
            f"{len(self.slice_artists)} time-slice lines | "
            + ("Surface mesh sampled for display; slice lines use all frequency bins. " if sampled else "Full-resolution surface. ")
            + "Shared amplitude scale; no per-slice normalization. Drag the plot to rotate."
        )
        self.toolbar.update()
        self.canvas.draw_idle()
