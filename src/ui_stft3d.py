"""Independent time-frequency surface with constant-time spectrum slices."""

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox,
    QPushButton, QSplitter, QScrollArea,
)
from matplotlib.figure import Figure
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from src.ui_components import SliderSpinBox
from src.surface_rendering import FrontLayerSurface
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
        self.opacity.setToolTip("Blend only the nearest surface layer with the background; hidden faces never add color.")
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
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        root.addWidget(scroll)
        graph = QWidget()
        layout = QVBoxLayout(graph)
        layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        root.addWidget(graph)
        root.setStretchFactor(0, 0)
        root.setStretchFactor(1, 1)
        root.setSizes([160, 600])
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
        self.redraw()

    def invalidate(self):
        """Clear outdated data after the processed source changes."""
        self.timer.stop()
        self._data = None
        self.redraw()
        self.caption.setText("Source changed. Click Show 3D Surface to refresh the spectrogram.")

    def reset_view(self):
        self._camera = (28, -58)
        if hasattr(self, "axis"):
            self.axis.view_init(elev=self._camera[0], azim=self._camera[1])
        self.redraw()

    def redraw(self):
        if hasattr(self, "axis"):
            self._camera = (self.axis.elev, self.axis.azim)
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
        self.surface = ScalarMappable(norm=Normalize(vmin=float(displayed.min()),
                                                     vmax=float(displayed.max())), cmap="viridis")
        self.surface.set_array(z)
        self.axis.auto_scale_xyz(x, y, z)
        if self.show_lines.isChecked():
            for index in self.slice_indices:
                # Full frequency resolution for each selected spectrum, no offsets.
                line, = self.axis.plot(np.full(len(frequencies), times[index]), frequencies,
                                       displayed[:, index], color="#142634", linewidth=0.85,
                                       alpha=1.0, zorder=3)
                self.slice_artists.append(line)
                # The depth-tested renderer draws only visible line fragments.
                line.set_visible(False)
        self.surface_artist = FrontLayerSurface(self.axis, x, y, z, self.surface,
                                                float(self.opacity.value()), self.slice_artists)
        self.axis.add_artist(self.surface_artist)
        self.axis.set_ylabel("Absolute frequency (Hz)" if folded else "Frequency (Hz)", labelpad=10)
        self.axis.set_zlabel(unit, labelpad=10)
        self.axis.set_title("STFT surface with constant-time spectrum slices")
        self.figure.colorbar(self.surface, ax=self.axis, shrink=0.65, pad=0.1, label=unit)
        sampled = len(freq_indices) < len(frequencies) or len(time_indices) < len(times)
        self.caption.setText(
            f"{len(times)} time frames × {len(frequencies)} frequency bins | "
            f"{len(self.slice_artists)} time-slice lines | "
            + ("Surface mesh sampled for display; slice lines use all frequency bins. " if sampled else "Full-resolution surface. ")
            + "Front-layer transparency; no stacked colors. Drag the plot to rotate."
        )
        self.toolbar.update()
        self.canvas.draw_idle()
