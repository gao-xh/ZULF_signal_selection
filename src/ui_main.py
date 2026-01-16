import sys
import numpy as np
import pandas as pd
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QGroupBox, QFileDialog, QSplitter, QProgressBar, QMessageBox,
    QTabWidget, QLabel, QListWidget, QAbstractItemView, QGridLayout, QDoubleSpinBox,
    QCheckBox
)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from src.ui_components import SliderSpinBox
from src.loader import ProgressiveLoader
from src.processing import Processor
from src.validator import SignalValidator
from src.config import UI_WINDOW_TITLE, UI_WINDOW_SIZE, UI_PARAM_RANGES

class LoaderWorker(QThread):
    finished = Signal(object, float, int) # avg_data, sampling_rate, scan_count
    error = Signal(str)

    def __init__(self, folder_paths):
        super().__init__()
        self.folder_paths = folder_paths

    def run(self):
        try:
            loader = ProgressiveLoader(self.folder_paths)
            count, avg_data = loader.get_full_average()
            if avg_data is None:
                self.error.emit("No valid data found.")
                return
            self.finished.emit(avg_data, loader.sampling_rate, count)
        except Exception as e:
            self.error.emit(str(e))

class ValidationWorker(QThread):
    finished = Signal(object, object, object, object) # results_df, freqs, spec, evo_data
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, folder_paths, params):
        super().__init__()
        self.folder_paths = folder_paths
        self.params = params

    def run(self):
        try:
            self.progress.emit("Loading Data...")
            loader = ProgressiveLoader(self.folder_paths)
            processor = Processor()
            validator = SignalValidator(loader, processor)
            
            self.progress.emit("Running Progressive Validation (This may take a while)...")
            results, freqs, spec, evo_data = validator.run_validation(self.params)
            
            self.finished.emit(results, freqs, spec, evo_data)
            
        except Exception as e:
            self.error.emit(str(e))

class ProcessWorker(QThread):
    finished = Signal(object, object, object) # freqs, spec, time_data
    error = Signal(str)
    
    def __init__(self, raw_data, params, sampling_rate):
        super().__init__()
        self.raw_data = raw_data
        self.params = params
        self.sampling_rate = sampling_rate

    def run(self):
        try:
            processor = Processor()
            # We modify process_fid to return time domain data too if needed, 
            # or we just rely on the fact that process_fid does the steps.
            # But wait, existing process_fid returns (freqs, spec). 
            # We might want the processed time domain signal for plotting.
            
            # Let's peek at processing.py again to see if we can get time data easily.
            # For now we'll just run it as is.
            freqs, spec = processor.process_fid(
                self.raw_data, 
                self.params,
                self.sampling_rate
            )
            # To get time data corresponding to this spectrum (processed), 
            # we rely on IFFT or we change the processor to return it. 
            # For visualization, IFFT of the spectrum is close enough to "processed result".
            processed_time = np.fft.ifft(spec)
            
            self.finished.emit(freqs, spec, processed_time)
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(UI_WINDOW_TITLE)
        self.resize(1400, 900) # Larger window for extra plot
        
        self.folder_paths = []
        
        # Data Model
        self.raw_avg_data = None # Holds the raw averaged (accumulated) FID
        self.loader_sampling_rate = 0.0
        
        self.current_results = None
        self.current_freqs = None
        self.current_spec = None
        self.current_evo_data = None
        self.current_processed_time = None
        
        # Debounce Timer for sliders
        self.update_timer = QTimer()
        self.update_timer.setSingleShot(True)
        self.update_timer.setInterval(300) # 300ms delay
        self.update_timer.timeout.connect(self.run_processing)
        
        self._setup_ui()
    
    def _unpack(self, range_tuple):
        # Config tuple: (min, max, step, default)
        # Needed for SliderSpinBox: (min, max, default, step)
        return range_tuple[0], range_tuple[1], range_tuple[3], range_tuple[2]
        
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        # --- Left Panel: Controls ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(300)
        
        # 0. Data Selection
        folder_group = QGroupBox("Data Selection")
        folder_layout = QVBoxLayout()
        self.folder_list = QListWidget()
        self.folder_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.folder_list.setFixedHeight(100)
        folder_layout.addWidget(self.folder_list)
        
        btn_layout = QHBoxLayout()
        self.btn_add_folder = QPushButton("Add Folder(s)")
        self.btn_add_folder.clicked.connect(self.add_folders)
        btn_layout.addWidget(self.btn_add_folder)
        
        self.btn_clear_folders = QPushButton("Clear")
        self.btn_clear_folders.clicked.connect(self.clear_folders)
        btn_layout.addWidget(self.btn_clear_folders)
        folder_layout.addLayout(btn_layout)
        folder_group.setLayout(folder_layout)
        left_layout.addWidget(folder_group)
        
        # 1. Processing Controls
        proc_group = QGroupBox("Processing Parameters")
        proc_layout = QVBoxLayout()
        
        r = UI_PARAM_RANGES
        
        # SVD Checkbox
        self.chk_svd = QCheckBox("Enable SVD Denoising")
        self.chk_svd.setChecked(True)
        self.chk_svd.setToolTip("Enable Singular Value Decomposition (Slow but effective)")
        proc_layout.addWidget(self.chk_svd)
        self.chk_svd.stateChanged.connect(self.request_processing_update)

        self.savgol_window = SliderSpinBox("Savgol Window", *self._unpack(r['savgol_window']))
        proc_layout.addWidget(self.savgol_window)
        # Connect signals
        self.savgol_window.valueChanged.connect(self.request_processing_update)
        
        self.savgol_order = SliderSpinBox("Savgol Order", *self._unpack(r['savgol_order']))
        proc_layout.addWidget(self.savgol_order)
        self.savgol_order.valueChanged.connect(self.request_processing_update)
        
        self.apod_rate = SliderSpinBox("Apod T2* (s)", *self._unpack(r['apod_t2star']), is_float=True, decimals=3)
        proc_layout.addWidget(self.apod_rate)
        self.apod_rate.valueChanged.connect(self.request_processing_update)

        # Truncation
        self.trunc_slider = SliderSpinBox("Trunc Start (pts)", *self._unpack(r['trunc_start']))
        proc_layout.addWidget(self.trunc_slider)
        self.trunc_slider.valueChanged.connect(self.request_processing_update)
        
        self.trunc_end_slider = SliderSpinBox("Trunc End (pts)", *self._unpack(r['trunc_end']))
        proc_layout.addWidget(self.trunc_end_slider)
        self.trunc_end_slider.valueChanged.connect(self.request_processing_update)

        # Fixed Phase
        self.p0_slider = SliderSpinBox("Phase 0 (deg)", *self._unpack(r['phase_0']), is_float=True)
        proc_layout.addWidget(self.p0_slider)
        self.p0_slider.valueChanged.connect(self.request_processing_update)

        self.p1_slider = SliderSpinBox("Phase 1 (deg)", *self._unpack(r['phase_1']), is_float=True)
        # Manually set a smaller visual step if needed, but config step 100 is good for large p1
        proc_layout.addWidget(self.p1_slider)
        self.p1_slider.valueChanged.connect(self.request_processing_update)
        
        proc_group.setLayout(proc_layout)
        left_layout.addWidget(proc_group)
        
        # 2. Validation Thresholds
        val_group = QGroupBox("Decision Thresholds")
        val_layout = QVBoxLayout()
        
        self.thr_r2 = SliderSpinBox("Min R2 Score", *self._unpack(r['min_r2']), is_float=True)
        val_layout.addWidget(self.thr_r2)
        self.thr_r2.valueChanged.connect(self.update_verdicts)
        
        self.thr_slope = SliderSpinBox("Min Slope", *self._unpack(r['min_slope']), is_float=True)
        val_layout.addWidget(self.thr_slope)
        self.thr_slope.valueChanged.connect(self.update_verdicts)
        
        val_group.setLayout(val_layout)
        left_layout.addWidget(val_group)
        
        # Action Buttons
        workflow_group = QGroupBox("Analysis Workflow")
        workflow_layout = QVBoxLayout()
        
        self.btn_load = QPushButton("1. Load Data")
        self.btn_load.clicked.connect(self.run_loading)
        workflow_layout.addWidget(self.btn_load)
        
        # Explicit Process button (in case SVD is heavy and auto-update is annoying)
        self.btn_reprocess = QPushButton("Refresh Processing")
        self.btn_reprocess.clicked.connect(self.run_processing)
        workflow_layout.addWidget(self.btn_reprocess)
        
        self.btn_run = QPushButton("2. Run Progressive Analysis")
        self.btn_run.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_run.clicked.connect(self.start_analysis)
        self.btn_run.setEnabled(False) 
        workflow_layout.addWidget(self.btn_run)
        
        workflow_group.setLayout(workflow_layout)
        left_layout.addWidget(workflow_group)
        
        self.progress_bar = QProgressBar()
        left_layout.addWidget(self.progress_bar)
        
        left_layout.addStretch()
        main_layout.addWidget(left_panel)
        
        # --- Right Panel: Visualization ---
        right_splitter = QSplitter(Qt.Vertical)
        
        # 0. Time Domain View (New)
        self.fig_time = Figure(figsize=(8, 3))
        self.canvas_time = FigureCanvas(self.fig_time)
        self.ax_time = self.fig_time.add_subplot(111)
        
        time_container = QWidget()
        time_layout = QVBoxLayout(time_container)
        time_layout.addWidget(NavigationToolbar(self.canvas_time, time_container))
        time_layout.addWidget(self.canvas_time)
        
        right_splitter.addWidget(time_container)
        
        # Top: Spectrum View
        self.fig_spec = Figure(figsize=(8, 4))
        self.canvas_spec = FigureCanvas(self.fig_spec)
        self.ax_spec = self.fig_spec.add_subplot(111)
        # Enable picking
        self.canvas_spec.mpl_connect('pick_event', self.on_pick)
        
        spec_container = QWidget()
        spec_layout = QVBoxLayout(spec_container)
        spec_layout.addWidget(NavigationToolbar(self.canvas_spec, spec_container))
        spec_layout.addWidget(self.canvas_spec)
        
        # Spectrum Controls (Range)
        range_group = QGroupBox("View Control")
        range_layout = QHBoxLayout()
        range_layout.setContentsMargins(5, 5, 5, 5)
        
        range_layout.addWidget(QLabel("Freq Range (Hz):"))
        
        self.freq_min = QDoubleSpinBox()
        self.freq_min.setRange(0, 1000)
        self.freq_min.setValue(0)
        self.freq_min.setSuffix(" Hz")
        range_layout.addWidget(self.freq_min)
        
        range_layout.addWidget(QLabel("-"))
        
        self.freq_max = QDoubleSpinBox()
        self.freq_max.setRange(0, 1000)
        self.freq_max.setValue(100)
        self.freq_max.setSuffix(" Hz")
        range_layout.addWidget(self.freq_max)
        
        btn_update_range = QPushButton("Update")
        btn_update_range.clicked.connect(self.plot_spectrum_traffic_light)
        range_layout.addWidget(btn_update_range)
        
        range_group.setLayout(range_layout)
        spec_layout.addWidget(range_group)

        # Spectrum View Mode (Real/Imag/Mag)
        view_mode_group = QGroupBox("Spectral Component")
        view_mode_layout = QHBoxLayout()
        view_mode_layout.setContentsMargins(5, 5, 5, 5)

        self.btn_view_mag = QPushButton("Magnitude")
        self.btn_view_mag.setCheckable(True)
        self.btn_view_mag.setChecked(True)
        self.btn_view_mag.clicked.connect(self.update_view_mode)
        view_mode_layout.addWidget(self.btn_view_mag)

        self.btn_view_real = QPushButton("Real")
        self.btn_view_real.setCheckable(True)
        self.btn_view_real.clicked.connect(self.update_view_mode)
        view_mode_layout.addWidget(self.btn_view_real)

        self.btn_view_imag = QPushButton("Imaginary")
        self.btn_view_imag.setCheckable(True)
        self.btn_view_imag.clicked.connect(self.update_view_mode)
        view_mode_layout.addWidget(self.btn_view_imag)
        
        # Exclusive selection logic is manual or use QButtonGroup, simplistic here:
        self.view_buttons = [self.btn_view_mag, self.btn_view_real, self.btn_view_imag]

        self.chk_view_abs = QCheckBox("Show Abs")
        self.chk_view_abs.setChecked(False)
        self.chk_view_abs.stateChanged.connect(self.update_view_mode)
        view_mode_layout.addWidget(self.chk_view_abs)

        view_mode_group.setLayout(view_mode_layout)
        spec_layout.addWidget(view_mode_group)

        right_splitter.addWidget(spec_container)
        
        # Bottom: Evolution View
        self.fig_evo = Figure(figsize=(8, 3))
        self.canvas_evo = FigureCanvas(self.fig_evo)
        self.ax_evo = self.fig_evo.add_subplot(111)
        
        evo_container = QWidget()
        evo_layout = QVBoxLayout(evo_container)
        evo_layout.addWidget(QLabel("Peak Evolution Analysis (Click a point above)"))
        evo_layout.addWidget(self.canvas_evo)
        right_splitter.addWidget(evo_container)
        
        main_layout.addWidget(right_splitter)

    def add_folders(self):
        # We allow multiple dirs. Qt QFileDialog.getExistingDirectory only does one.
        # But we can call it multiple times or use a workaround.
        # Standard approach: Call valid folder. User can add more.
        folder = QFileDialog.getExistingDirectory(self, "Select Experiment Folder")
        if folder:
            path = str(Path(folder).resolve())
            if path not in self.folder_paths:
                self.folder_paths.append(path)
                self.folder_list.addItem(path)

    def clear_folders(self):
        self.folder_paths = []
        self.folder_list.clear()
        self.btn_run.setEnabled(False)

    def _get_process_params(self):
        # Determine current detection mode
        detect_mode = 'mag'
        if self.btn_view_real.isChecked():
            detect_mode = 'real'
        elif self.btn_view_imag.isChecked():
            detect_mode = 'imag'
            
        return {
            'conv_points': int(self.savgol_window.value()) if int(self.savgol_window.value()) % 2 != 0 else int(self.savgol_window.value()) + 1,
            'poly_order': int(self.savgol_order.value()),
            'apod_t2star': float(self.apod_rate.value()),
            'p0': float(self.p0_slider.value()),
            'p1': float(self.p1_slider.value()),
            'trunc_start': int(self.trunc_slider.value()),
            'trunc_end': int(self.trunc_end_slider.value()),
            'phase_mode': 'manual', 
            'enable_svd': self.chk_svd.isChecked(),
            'detect_mode': detect_mode # Pass to worker -> validator
        }

    def request_processing_update(self):
        if self.raw_avg_data is not None:
            self.update_timer.start()

    def run_loading(self):
        if not self.folder_paths:
            QMessageBox.warning(self, "No Data", "Please add at least one folder.")
            return

        self.btn_load.setEnabled(False)
        self.btn_reprocess.setEnabled(False)
        self.btn_run.setEnabled(False)
        self.progress_bar.setRange(0, 0) # Indeterminate
        self.statusBar().showMessage("Loading and Averaging Data...")
        
        self.loader_worker = LoaderWorker(self.folder_paths)
        self.loader_worker.finished.connect(self.on_loading_finished)
        self.loader_worker.error.connect(self.on_error)
        self.loader_worker.start()

    def on_loading_finished(self, avg_data, sampling_rate, count):
        self.raw_avg_data = avg_data
        self.loader_sampling_rate = sampling_rate
        
        self.btn_load.setEnabled(True)
        self.btn_reprocess.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.statusBar().showMessage(f"Loaded {count} scans. Running initial processing...")
        
        # Auto trigger processing
        self.run_processing()

    def run_processing(self):
        if self.raw_avg_data is None:
            return

        params = self._get_process_params()
        self.statusBar().showMessage("Processing...")
        
        self.proc_worker = ProcessWorker(self.raw_avg_data, params, self.loader_sampling_rate)
        self.proc_worker.finished.connect(self.on_processing_finished)
        self.proc_worker.error.connect(self.on_error)
        self.proc_worker.start()

    def on_processing_finished(self, freqs, spec, time_data):
        self.current_freqs = freqs
        self.current_spec = spec
        self.current_processed_time = time_data
        
        self.btn_run.setEnabled(True)
        self.statusBar().showMessage("Processing Complete.")
        
        self.plot_time_domain()
        self.plot_spectrum_traffic_light()

    def plot_time_domain(self):
        self.ax_time.clear()
        if self.current_processed_time is None:
            self.canvas_time.draw()
            return
            
        # Plot Real part of time domain usually
        t = np.arange(len(self.current_processed_time)) / self.loader_sampling_rate
        y = np.real(self.current_processed_time)
        
        self.ax_time.plot(t, y, 'b-', linewidth=0.5)
        self.ax_time.set_title("Time Domain Signal (Processed)")
        self.ax_time.set_xlabel("Time (s)")
        self.ax_time.set_ylabel("Amplitude")
        self.ax_time.grid(True, linestyle='--', alpha=0.5)

        self.fig_time.tight_layout()
        self.canvas_time.draw()
        self.worker.start()
        
    def update_status(self, msg):
        self.statusBar().showMessage(msg)
        
    def on_error(self, msg):
        self.btn_run.setEnabled(True)
        self.btn_preview.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.statusBar().showMessage("Error")
        QMessageBox.critical(self, "Analysis Failed", msg)
        
    def on_analysis_finished(self, results, freqs, spec, evo_data):
        self.btn_run.setEnabled(True)
        self.btn_preview.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.statusBar().showMessage("Analysis Complete")
        
        self.current_results = results
        self.current_freqs = freqs
        self.current_spec = spec
        self.current_evo_data = evo_data
        
        # Populate UI
        if 'p0' in self.worker.params:
            self.p0_slider.blockSignals(True)
            self.p0_slider.spinbox.setValue(float(self.worker.params.get('p0', 0)))
            self.p0_slider.blockSignals(False)

        self.update_verdicts() # This triggers plotting
        
    def update_verdicts(self):
        if self.current_results is None:
            return
            
        r2_thr = self.thr_r2.value()
        slope_thr = self.thr_slope.value()
        
        # Re-evaluate verdicts based on new sliders
        def judge(row):
            return (row['R2'] >= r2_thr) and (row['Slope'] >= slope_thr)
            
        self.current_results['Verdict'] = self.current_results.apply(judge, axis=1)
        self.plot_spectrum_traffic_light()
        
    def update_view_mode(self):
        sender = self.sender()
        if not sender: return
        
        # Enforce exclusivity
        for btn in self.view_buttons:
            if btn != sender:
                btn.setChecked(False)
        sender.setChecked(True)
        
        self.plot_spectrum_traffic_light()

    def plot_spectrum_traffic_light(self):
        self.ax_spec.clear()
        
        if self.current_freqs is None or self.current_spec is None:
            self.canvas_spec.draw()
            return
        
        # Determine data to plot based on view mode
        spec_data = self.current_spec
        mode_label = "Magnitude"
        
        if self.btn_view_real.isChecked():
            # Usually we want Real component for Phasing
            plot_data = np.real(spec_data)
            mode_label = "Real"
        elif self.btn_view_imag.isChecked():
            plot_data = np.imag(spec_data)
            mode_label = "Imaginary"
        else:
            # Default Magnitude
            plot_data = np.abs(spec_data)
            mode_label = "Magnitude"

        # Apply absolute value if requested
        if self.chk_view_abs.isChecked():
            plot_data = np.abs(plot_data)
            mode_label += " (Abs)"

        # Plot Spectrum (Positive frequencies only)
        freqs = self.current_freqs
        
        # Mask for positive frequencies
        pos_mask = freqs >= 0
        pos_freqs = freqs[pos_mask]
        pos_data = plot_data[pos_mask]
        
        self.ax_spec.plot(pos_freqs, pos_data, 'k-', linewidth=0.8, alpha=0.6, label=f'Spectrum ({mode_label})')
        
        # Plot Candidates (Overlaid on the selected view)
        # Note: Peak picking is usually done on Magnitude, so the Y-positions might need adjustment if viewing Real.
        # But for visualization context, we can plot the marker at the Y-value of the CURRENT view.
        
        df = self.current_results
        
        if df is not None:
            # Filter candidates for positive frequencies
            df_pos = df[df['Freq_Hz'] >= 0]
            
            signals = df_pos[df_pos['Verdict'] == True]
            noise = df_pos[df_pos['Verdict'] == False]
            
            # Helper to get Y-coord from current plot data for specific indices
            # Index in df maps to full spectrum, need to map to pos_mask if we used it...
            # Actually easier: just use the index directly on plot_data, then filter out neg freqs
            
            def get_y_values(indices):
                # indices are integers into the full array
                return plot_data[indices.astype(int)]

            # Plot Green Signals
            if not signals.empty:
                self.ax_spec.scatter(
                    signals['Freq_Hz'], 
                    get_y_values(signals['Index']), 
                    c='g', s=100, marker='o', alpha=0.8, edgecolors='k', 
                    label='Signal (Validated)', picker=5
                )
                
            # Plot Red Noise
            if not noise.empty:
                self.ax_spec.scatter(
                    noise['Freq_Hz'], 
                    get_y_values(noise['Index']), 
                    c='r', s=50, marker='x', alpha=0.6, 
                    label='Noise', picker=5
                )
        
        # Adjust X-Axis to show only positive part or user selected range
        x_min = self.freq_min.value()
        x_max = self.freq_max.value()
        
        # If default 0-100 is untouched and data goes further, maybe auto-scale?
        # But user wants control. Let's respect the spinboxes.
        self.ax_spec.set_xlim(left=x_min, right=x_max)
        
        # Auto-scale Y based on visible X range
        if len(pos_freqs) > 0:
            mask = (pos_freqs >= x_min) & (pos_freqs <= x_max)
            if np.any(mask):
                y_visible = pos_data[mask]
                if len(y_visible) > 0:
                    y_max = np.max(y_visible)
                    y_min = np.min(y_visible)
                    
                    # Add some padding
                    margin = (y_max - y_min) * 0.1
                    if margin == 0: margin = 1
                    self.ax_spec.set_ylim(bottom=y_min - margin, top=y_max + margin)

        self.ax_spec.set_title("Spectral Validation" + (" (Preview)" if df is None else " (Click points for details)"))
        self.ax_spec.set_xlabel("Frequency (Hz)")
        self.ax_spec.set_ylabel(f"Amplitude ({mode_label})")
        self.ax_spec.legend(loc='upper right')
        self.fig_spec.tight_layout()
        self.canvas_spec.draw()
        
    def on_pick(self, event):
        ind = event.ind[0] # Index within the collection (scatter plot)
        # Identify which collection was clicked
        # This is tricky with multiple scatters.
        # Simplified: We find the closest candidate to the click X coordinate
        
        click_x = event.mouseevent.xdata
        if click_x is None: return
        
        # Find closest candidate in dataframe
        df = self.current_results
        # Calculate distance
        df['dist'] = abs(df['Freq_Hz'] - click_x)
        closest_row = df.loc[df['dist'].idxmin()]
        
        # If reasonably close (e.g. 1Hz)
        if closest_row['dist'] < 5.0:
            self.plot_evolution(closest_row['Index'])

    def plot_evolution(self, idx):
        self.ax_evo.clear()
        
        if self.current_evo_data is None or idx not in self.current_evo_data:
            return
            
        data = self.current_evo_data[idx]
        x = np.array(data['sqrt_N'])
        y = np.array(data['SNR'])
        
        # Plot Scatter
        self.ax_evo.scatter(x, y, c='b', alpha=0.6, label='Measurements')
        
        # Plot Regression Line
        # We need to recalc or store slope/intercept. Recalc is cheap for a few points.
        from scipy.stats import linregress
        if len(x) > 1:
            slope, intercept, r_val, _, _ = linregress(x, y)
            y_pred = slope * x + intercept
            
            # Color code based on if it passes "good" criteria
            color = 'g' if slope > 0.1 and r_val**2 > 0.8 else 'r'
            self.ax_evo.plot(x, y_pred, '-', color=color, linewidth=2, label=f'Fit (R2={r_val**2:.2f})')
            
        self.ax_evo.set_xlabel("sqrt(N) [Scaling Factor]")
        self.ax_evo.set_ylabel("SNR")
        self.ax_evo.set_title(f"Signal Evolution: Freq Index {idx}")
        self.ax_evo.grid(True, linestyle='--', alpha=0.5)
        self.ax_evo.legend()
        
        self.fig_evo.tight_layout()
        self.canvas_evo.draw()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
