import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QGroupBox, QFileDialog, QSplitter, QProgressBar, QMessageBox,
    QTabWidget, QLabel, QListWidget, QAbstractItemView, QGridLayout, QDoubleSpinBox,
    QCheckBox, QComboBox, QScrollArea, QSpinBox, QSizePolicy
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

class RelaxationWorker(QThread):
    finished = Signal(object, object, object, object, float, float) # times, amps, fit_x, fit_y, t2, r2
    progress = Signal(str, int) # message, percent
    error = Signal(str)

    def __init__(self, full_data, target_freq, sampling_rate, params, t_start=0, t_end=0, points=50, zero_fill_front=False):
        super().__init__()
        self.data = full_data
        self.target_freq = target_freq
        self.fs = sampling_rate
        self.params = params
        self.t_start = t_start
        self.t_end = t_end
        self.points = points
        self.zero_fill_front = zero_fill_front

    def run(self):
        try:
            from scipy.stats import linregress
            import scipy.fft
            
            # Setup sweep parameters
            N = len(self.data)
            
            # Use user provided range if valid, else defaults
            if self.t_end > self.t_start:
                start_p = int(self.t_start * self.fs)
                end_p = int(self.t_end * self.fs)
            else:
                # Fallback to logic: 0 to 60%
                start_p = 0
                end_p = int(N * 0.6) 

            # Validation
            if start_p < 0: start_p = 0
            if end_p >= N: end_p = N - 1
            if end_p <= start_p: 
                end_p = start_p + 100 # Minimum fallback
            
            # Determine Step size 
            span = end_p - start_p
            step = max(1, span // self.points)
            
            trunc_points = range(start_p, end_p, step)
            
            times = []
            amps = []
            
            # Pre-calculate window for peak search
            full_freqs = scipy.fft.fftfreq(N, d=1.0/self.fs)
            target_idx = np.argmin(np.abs(full_freqs - self.target_freq))
            search_r = 5 
            
            # Helper to apply processing on segment
            def apply_segment_processing(segment_data):
                # 1. Savgol (Baseline)
                if self.params.get('conv_points', 0) > 0:
                    window = int(self.params['conv_points'])
                    order = int(self.params['poly_order'])
                    if window % 2 == 0: window += 1
                    if window > 3 and window < len(segment_data):
                        smooth = scipy.signal.savgol_filter(segment_data.real, window, order, mode='mirror')
                        segment_data = segment_data - smooth
                
                # 2. Apodization
                # Note: Apodization usually starts from t=0. 
                # Here t=0 corresponds to trunc_start relative to original signal.
                # If we want to clean up the 'new' signal, we apodize from its start.
                t2_apod = self.params.get('apod_t2star', 0)
                if t2_apod > 0:
                     t_axis = np.linspace(0, len(segment_data)/self.fs, len(segment_data), endpoint=False)
                     window_func = np.exp(-t_axis / t2_apod)
                     segment_data = segment_data * window_func
                     
                return segment_data

            # Optimization: Pre-allocate buffer for FFT to avoid reallocation in loop
            # Use complex128 if input is complex, else float
            dtype = self.data.dtype if np.iscomplexobj(self.data) else np.complex128
            padded = np.zeros(N, dtype=dtype)
            
            total_steps = len(trunc_points)
            for i, start_iter in enumerate(trunc_points):
                # Report Progress
                if i % 5 == 0:
                    pcl = int((i / total_steps) * 100)
                    self.progress.emit(f"Analyzing Relaxation Step {i}/{total_steps}...", pcl)

                # Extract segment
                raw_segment = self.data[start_iter:]
                
                # Apply Processing (Baseline, Appodization)
                processed_segment = apply_segment_processing(raw_segment)
                
                if self.zero_fill_front:
                    # MODE: Zero-Fill Front (Keep data in place)
                    # Structure: [0...0, processed_segment..., 0...]
                    # Clear buffer to zero (in case it has old data)
                    padded[:] = 0
                    # Insert at original position
                    # processed_segment length is (N - start_iter)
                    padded[start_iter:] = processed_segment
                else:
                    # MODE: Shift Left (Cut front) - Standard
                    # Structure: [processed_segment..., 0...0]
                    # Copy to start
                    padded[:len(processed_segment)] = processed_segment
                    # Clean the rest
                    padded[len(processed_segment):] = 0
                
                # 3. FFT
                # overwrite_x=False to keep padded clean? No, we rewrite it anyway.

                # overwrite_x=False to keep padded clean? No, we rewrite it anyway.
                # But standard FFT might allocate new output array.
                spec = scipy.fft.fft(padded)
                spec_mag = np.abs(spec)
                
                # 4. Measure
                idx_start = max(0, target_idx - search_r)
                idx_end = min(N, target_idx + search_r + 1)
                
                if idx_end > idx_start:
                    peak_amp = np.max(spec_mag[idx_start:idx_end])
                else:
                    peak_amp = 0
                
                times.append(start_iter / self.fs)
                amps.append(peak_amp)
                
            times = np.array(times)
            amps = np.array(amps)
            
            # 5. Fit T2*
            valid = amps > 0
            if np.sum(valid) > 2:
                x_fit = times[valid]
                y_fit_log = np.log(amps[valid])
                
                slope, intercept, r_val, _, _ = linregress(x_fit, y_fit_log)
                
                if slope < 0:
                    t2 = -1.0 / slope
                else:
                    t2 = 0 
                
                r2 = r_val**2
                
                if r2 > 0:
                     fit_x = np.linspace(min(x_fit), max(x_fit), 50)
                     fit_y = np.exp(intercept + slope * fit_x)
                else:
                     fit_x = []
                     fit_y = []
            else:
                fit_x = []
                fit_y = []
                t2 = 0
                r2 = 0
                
            self.finished.emit(times, amps, fit_x, fit_y, t2, r2)
            
        except Exception as e:
            self.error.emit(str(e))


class BatchRelaxationWorker(QThread):
    finished = Signal(object) # results_dict
    progress = Signal(str, int)
    error = Signal(str)

    def __init__(self, full_data, target_freqs_df, sampling_rate, params, t_start=0, t_end=0, points=50, zero_fill_front=False, use_tracking=False, track_win_hz=5.0):
        super().__init__()
        self.data = full_data
        self.target_freqs = target_freqs_df # DataFrame with 'Freq_Hz' column
        self.fs = sampling_rate
        self.params = params
        self.t_start = t_start
        self.t_end = t_end
        self.points = points
        self.zero_fill_front = zero_fill_front
        self.use_tracking = use_tracking
        self.track_win_hz = track_win_hz

    def run(self):
        try:
            from scipy.stats import linregress
            import scipy.fft
            
            # Use complex128 if input is complex, else float
            dtype = self.data.dtype if np.iscomplexobj(self.data) else np.complex128
            N = len(self.data)
            full_freqs = scipy.fft.fftfreq(N, d=1.0/self.fs)
            
            # Param Ranges
            if self.t_end > self.t_start:
                start_p = int(self.t_start * self.fs)
                end_p = int(self.t_end * self.fs)
            else:
                start_p = 0
                end_p = int(N * 0.6)
                
            if start_p < 0: start_p = 0
            if end_p >= N: end_p = N - 1
            if end_p <= start_p: end_p = start_p + 100
            
            span = end_p - start_p
            step = max(1, span // self.points)
            trunc_points_list = list(range(start_p, end_p, step))
            
            # Helper Logic (Duplicated from RelaxationWorker to be self-contained in thread)
            def apply_segment_processing(segment_data):
                # 1. Savgol 
                if self.params.get('conv_points', 0) > 0:
                    window = int(self.params['conv_points'])
                    order = int(self.params['poly_order'])
                    if window % 2 == 0: window += 1
                    if window > 3 and window < len(segment_data):
                        smooth = scipy.signal.savgol_filter(segment_data.real, window, order, mode='mirror')
                        segment_data = segment_data - smooth
                # 2. Apodization
                t2_apod = self.params.get('apod_t2star', 0)
                if t2_apod > 0:
                     t_axis = np.linspace(0, len(segment_data)/self.fs, len(segment_data), endpoint=False)
                     window_func = np.exp(-t_axis / t2_apod)
                     segment_data = segment_data * window_func
                return segment_data
            
            # Pre-allocate buffer
            padded = np.zeros(N, dtype=dtype)
            
            # Prepare result structures
            # Dictionary: freq -> {times:[], amps:[]}
            curve_data = { row['Freq_Hz']: {'times': [], 'amps': []} for _, row in self.target_freqs.iterrows() }
            
            # Map freq to index in FFT
            freq_to_idx = {}
            for f in curve_data.keys():
                freq_to_idx[f] = np.argmin(np.abs(full_freqs - f))
            
            # Define State for Iterative Tracking
            current_peak_indices = freq_to_idx.copy()
            
            # Calculate Search Radius
            hz_per_point = self.fs / N if N > 0 else 1.0
            if self.use_tracking and self.track_win_hz > 0:
                # Radius = Half Window / Resolution
                r_pts = int(np.ceil((self.track_win_hz / 2.0) / hz_per_point))
                search_r = max(1, r_pts)
            else:
                search_r = 5 # Default static small window

            total_steps = len(trunc_points_list)
            
            for i, start_iter in enumerate(trunc_points_list):
                 if i % 5 == 0:
                    pcl = int((i / total_steps) * 100)
                    self.progress.emit(f"Batch Analysis: Scanning Time Step {i}/{total_steps}...", pcl)
                    
                 # 1. Processing
                 raw_segment = self.data[start_iter:]
                 processed_segment = apply_segment_processing(raw_segment)
                 
                 if self.zero_fill_front:
                    padded[:] = 0
                    padded[start_iter:] = processed_segment
                 else:
                    padded[:len(processed_segment)] = processed_segment
                    padded[len(processed_segment):] = 0
                    
                 # 2. FFT
                 spec = scipy.fft.fft(padded)
                 spec_mag = np.abs(spec)
                 
                 current_time = start_iter / self.fs
                 
                 # Prepare next iteration updates
                 next_indices_update = {}

                 # 3. Measure All Peaks
                 for f_key in curve_data.keys():
                     # Use current tracked index
                     idx_center = current_peak_indices[f_key]
                     
                     idx_start = max(0, idx_center - search_r)
                     idx_end = min(N, idx_center + search_r + 1)
                     
                     if idx_end > idx_start:
                        segment_view = spec_mag[idx_start:idx_end]
                        peak_amp = np.max(segment_view)
                        
                        # Logic: If tracking, find local max index to update center
                        if self.use_tracking:
                            local_max = np.argmax(segment_view)
                            abs_max_idx = idx_start + local_max
                            next_indices_update[f_key] = abs_max_idx
                     else:
                        peak_amp = 0
                        if self.use_tracking:
                            next_indices_update[f_key] = idx_center # Keep if lost
                     
                     curve_data[f_key]['times'].append(current_time)
                     curve_data[f_key]['amps'].append(peak_amp)
                
                 # Update centers for next step
                 if self.use_tracking:
                     current_peak_indices.update(next_indices_update)
            
            # Post-processing: Fit T2* for all
            summary_list = []
            
            detailed_results = {}
            
            for f_key, data_c in curve_data.items():
                 times = np.array(data_c['times'])
                 amps = np.array(data_c['amps'])
                 
                 t2 = 0
                 r2 = 0
                 fit_x = []
                 fit_y = []
                 
                 valid = amps > 0
                 if np.sum(valid) > 2:
                     x_fit = times[valid]
                     y_fit_log = np.log(amps[valid])
                     slope, intercept, r_val, _, _ = linregress(x_fit, y_fit_log)
                     
                     if slope < 0:
                        t2 = -1.0 / slope
                     r2 = r_val**2
                     
                     if r2 > 0:
                         fit_x = np.linspace(min(x_fit), max(x_fit), 50)
                         fit_y = np.exp(intercept + slope * fit_x)
                 
                 summary_list.append({
                     'Freq_Hz': f_key,
                     'T2_star_ms': t2 * 1000.0,
                     'R2': r2
                 })
                 
                 detailed_results[f_key] = {
                     'times': times,
                     'amps': amps,
                     'fit_x': fit_x,
                     'fit_y': fit_y,
                     't2': t2,
                     'r2': r2
                 }
                 
            summary_df = pd.DataFrame(summary_list)
            
            self.finished.emit({
                'summary': summary_df,
                'details': detailed_results
            })
            
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
        
        self.batch_results_summary = None
        self.batch_results_details = None

        # New: Explicit sampling rate storage
        self.sampling_rate = 1000.0 # Default fallback

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
        # Menu Bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        
        load_action = file_menu.addAction("Load Parameters")
        load_action.triggered.connect(self.load_parameters)
        
        save_action = file_menu.addAction("Save Parameters")
        save_action.triggered.connect(self.save_parameters)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Main Splitter: Left (Controls) | Right (Plots)
        self.main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self.main_splitter)
        
        # --- Left Panel: Controls (Scrollable) ---
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QScrollArea.NoFrame)
        left_scroll.setMinimumWidth(390) 
        
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(5, 5, 5, 5)
        
        left_scroll.setWidget(left_panel)
        self.main_splitter.addWidget(left_scroll)
        
        # 0. Data Selection (Common)
        folder_group = QGroupBox("Data Selection")
        folder_layout = QVBoxLayout()
        folder_layout.setContentsMargins(5, 5, 5, 5) # Compact margins
        
        self.folder_list = QListWidget()
        self.folder_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.folder_list.setFixedHeight(80)
        folder_layout.addWidget(self.folder_list)
        
        btn_layout = QHBoxLayout()
        self.btn_add_folder = QPushButton("Add")
        self.btn_add_folder.clicked.connect(self.add_folders)
        btn_layout.addWidget(self.btn_add_folder)
        
        self.btn_clear_folders = QPushButton("Clear")
        self.btn_clear_folders.clicked.connect(self.clear_folders)
        btn_layout.addWidget(self.btn_clear_folders)
        folder_layout.addLayout(btn_layout)

        self.btn_load = QPushButton("Load Data")
        self.btn_load.clicked.connect(self.run_loading)
        folder_layout.addWidget(self.btn_load)

        folder_group.setLayout(folder_layout)
        # Prevent it from expanding vertically
        folder_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        left_layout.addWidget(folder_group)
        
        # --- Tab ToolBox ---
        self.tabs_control = QTabWidget()
        left_layout.addWidget(self.tabs_control)

        # === TAB 1: PREPROCESSING ===
        self.tab_process = QWidget()
        proc_tab_layout = QVBoxLayout(self.tab_process)
        proc_tab_layout.setContentsMargins(5, 5, 5, 5)

        # Signal Processing Params
        r = UI_PARAM_RANGES
        proc_group = QGroupBox("Signal Parameters")
        proc_layout = QVBoxLayout()
        
        # SVD
        self.chk_svd = QCheckBox("Enable SVD Denoising")
        self.chk_svd.setToolTip("Enable Singular Value Decomposition")
        proc_layout.addWidget(self.chk_svd)
        self.chk_svd.stateChanged.connect(self.request_processing_update)

        # Savgol
        self.savgol_window = SliderSpinBox("Savgol Window", *self._unpack(r['savgol_window']))
        proc_layout.addWidget(self.savgol_window)
        self.savgol_window.valueChanged.connect(self.request_processing_update)
        
        self.savgol_order = SliderSpinBox("Savgol Order", *self._unpack(r['savgol_order']))
        proc_layout.addWidget(self.savgol_order)
        self.savgol_order.valueChanged.connect(self.request_processing_update)
        
        # Apod & Trunc
        self.apod_rate = SliderSpinBox("Apod T2* (s)", *self._unpack(r['apod_t2star']), is_float=True, decimals=3)
        proc_layout.addWidget(self.apod_rate)
        self.apod_rate.valueChanged.connect(self.request_processing_update)

        self.trunc_slider = SliderSpinBox("Trunc Start (pts)", *self._unpack(r['trunc_start']))
        proc_layout.addWidget(self.trunc_slider)
        self.trunc_slider.valueChanged.connect(self.request_processing_update)
        
        self.trunc_end_slider = SliderSpinBox("Trunc End (pts)", *self._unpack(r['trunc_end']))
        proc_layout.addWidget(self.trunc_end_slider)
        self.trunc_end_slider.valueChanged.connect(self.request_processing_update)

        # Phase
        self.p0_slider = SliderSpinBox("Phase 0 (deg)", *self._unpack(r['phase_0']), is_float=True)
        proc_layout.addWidget(self.p0_slider)
        self.p0_slider.valueChanged.connect(self.request_processing_update)

        self.p1_slider = SliderSpinBox("Phase 1 (deg)", *self._unpack(r['phase_1']), is_float=True)
        proc_layout.addWidget(self.p1_slider)
        self.p1_slider.valueChanged.connect(self.request_processing_update)
        
        proc_group.setLayout(proc_layout)
        proc_tab_layout.addWidget(proc_group)
        
        # Bottom Button
        self.btn_reprocess = QPushButton("Refresh Processing")
        self.btn_reprocess.clicked.connect(self.run_processing)
        proc_tab_layout.addWidget(self.btn_reprocess)
        proc_tab_layout.addStretch()
        
        self.tabs_control.addTab(self.tab_process, "Processing")


        # === TAB 2: DETECTION ===
        self.tab_detect = QWidget()
        det_tab_layout = QVBoxLayout(self.tab_detect)
        det_tab_layout.setContentsMargins(5, 5, 5, 5)
        
        # Peak Detection Settings
        peak_group = QGroupBox("Peak Detection")
        peak_layout = QVBoxLayout()
        
        self.peak_thr = SliderSpinBox("Min Abs Height", *self._unpack(r['peak_height']), is_float=True)
        peak_layout.addWidget(self.peak_thr)
        self.peak_thr.valueChanged.connect(self.plot_spectrum_traffic_light)

        self.freq_min_search = SliderSpinBox("Search Freq Min", *self._unpack(r['search_freq_min']), is_float=True)
        peak_layout.addWidget(self.freq_min_search)
        self.freq_min_search.valueChanged.connect(self.plot_spectrum_traffic_light)

        self.freq_max_search = SliderSpinBox("Search Freq Max", *self._unpack(r['search_freq_max']), is_float=True)
        peak_layout.addWidget(self.freq_max_search)
        self.freq_max_search.valueChanged.connect(self.plot_spectrum_traffic_light)
        
        self.peak_win = SliderSpinBox("Search Range (pts)", *self._unpack(r['peak_window']))
        peak_layout.addWidget(self.peak_win)

        # Noise Method
        noise_method_layout = QHBoxLayout()
        noise_method_layout.addWidget(QLabel("Noise Method:"))
        self.combo_noise_method = QComboBox()
        self.combo_noise_method.addItems(["Global Region", "Local Window"])
        self.combo_noise_method.currentTextChanged.connect(self.update_noise_ui_visibility)
        noise_method_layout.addWidget(self.combo_noise_method)
        peak_layout.addLayout(noise_method_layout)

        # Global Region
        self.noise_global_group = QWidget()
        n_glob_l = QVBoxLayout(self.noise_global_group)
        n_glob_l.setContentsMargins(0,0,0,0)
        self.noise_min = SliderSpinBox("Noise Min (Hz)", *self._unpack(r['noise_freq_min']), is_float=True)
        self.noise_min.valueChanged.connect(self.plot_spectrum_traffic_light)
        n_glob_l.addWidget(self.noise_min)

        self.noise_max = SliderSpinBox("Noise Max (Hz)", *self._unpack(r['noise_freq_max']), is_float=True)
        self.noise_max.valueChanged.connect(self.plot_spectrum_traffic_light)
        n_glob_l.addWidget(self.noise_max)
        peak_layout.addWidget(self.noise_global_group)

        # Local Window
        self.noise_local_group = QWidget()
        n_loc_l = QVBoxLayout(self.noise_local_group)
        n_loc_l.setContentsMargins(0,0,0,0)
        self.noise_local_win = SliderSpinBox("Local Window", *self._unpack(r['local_noise_window']))
        n_loc_l.addWidget(self.noise_local_win)
        peak_layout.addWidget(self.noise_local_group)
        self.noise_local_group.setVisible(False)

        peak_group.setLayout(peak_layout)
        det_tab_layout.addWidget(peak_group)

        # Decision Thresholds
        val_group = QGroupBox("Decision Thresholds")
        val_layout = QVBoxLayout()
        self.thr_r2 = SliderSpinBox("Min R2 Score", *self._unpack(r['min_r2']), is_float=True)
        val_layout.addWidget(self.thr_r2)
        self.thr_r2.valueChanged.connect(self.update_verdicts)
        
        self.thr_slope = SliderSpinBox("Min Slope", *self._unpack(r['min_slope']), is_float=True, decimals=3)
        val_layout.addWidget(self.thr_slope)
        self.thr_slope.valueChanged.connect(self.update_verdicts)
        val_group.setLayout(val_layout)
        det_tab_layout.addWidget(val_group)
        
        # Bottom Button
        self.btn_run = QPushButton("Run Progressive Analysis")
        self.btn_run.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_run.clicked.connect(self.start_analysis)
        self.btn_run.setEnabled(False) 
        det_tab_layout.addWidget(self.btn_run)
        
        det_tab_layout.addStretch()
        self.tabs_control.addTab(self.tab_detect, "Detection")


        # === TAB 3: ANALYSIS ===
        self.tab_analysis = QWidget()
        ana_tab_layout = QVBoxLayout(self.tab_analysis)
        ana_tab_layout.setContentsMargins(5, 5, 5, 5)

        # Mode
        mode_group = QGroupBox("Mode Selection")
        mode_l = QVBoxLayout()
        self.lbl_analysis_mode = QLabel("Analysis Task:")
        self.combo_analysis_mode = QComboBox()
        self.combo_analysis_mode.addItems(["Signal Evolution (SNR vs N)", "Relaxation Analysis (Amp vs t)", "Global T2* Map"])
        self.combo_analysis_mode.currentIndexChanged.connect(self.on_analysis_mode_changed)
        mode_l.addWidget(self.lbl_analysis_mode)
        mode_l.addWidget(self.combo_analysis_mode)
        mode_group.setLayout(mode_l)
        ana_tab_layout.addWidget(mode_group)

        # Relaxation Params (Container)
        self.relax_settings_widget = QGroupBox("Relaxation Parameters")
        relax_layout = QVBoxLayout()
        
        unit_layout = QHBoxLayout()
        unit_layout.addWidget(QLabel("Unit:"))
        self.combo_relax_unit = QComboBox()
        self.combo_relax_unit.addItems(["Time (ms)", "Points"])
        self.combo_relax_unit.currentTextChanged.connect(self.update_relax_ui_state)
        unit_layout.addWidget(self.combo_relax_unit)
        relax_layout.addLayout(unit_layout)

        self.lbl_relax_start = QLabel("Start:")
        relax_layout.addWidget(self.lbl_relax_start)
        self.spin_relax_start = QDoubleSpinBox()
        self.spin_relax_start.setRange(0, 500000) 
        self.spin_relax_start.setValue(0)
        self.spin_relax_start.setSingleStep(10)
        self.spin_relax_start.setSuffix(" ms")
        relax_layout.addWidget(self.spin_relax_start)
        
        self.lbl_relax_end = QLabel("End:")
        relax_layout.addWidget(self.lbl_relax_end)
        self.spin_relax_end = QDoubleSpinBox()
        self.spin_relax_end.setRange(0, 500000)
        self.spin_relax_end.setValue(500) 
        self.spin_relax_end.setSingleStep(10)
        self.spin_relax_end.setSuffix(" ms")
        relax_layout.addWidget(self.spin_relax_end)
        
        pts_layout = QHBoxLayout()
        pts_layout.addWidget(QLabel("Step Points:"))
        self.spin_relax_points = QSpinBox()
        self.spin_relax_points.setRange(10, 500)
        self.spin_relax_points.setValue(50)
        self.spin_relax_points.setSingleStep(10)
        pts_layout.addWidget(self.spin_relax_points)
        relax_layout.addLayout(pts_layout)
        
        self.chk_relax_zerofill = QCheckBox("Zero-Fill Front")
        self.chk_relax_zerofill.setToolTip("Truncated points renamed to zero")
        relax_layout.addWidget(self.chk_relax_zerofill)

        # -- Iterative Tracking Controls --
        tracking_layout = QHBoxLayout()
        self.chk_iterative_tracking = QCheckBox("Iterative Track")
        self.chk_iterative_tracking.setToolTip("Dynamic center search (Center(t) = Peak(t-1))")
        # Connect to sync logic
        self.chk_iterative_tracking.stateChanged.connect(self.sync_iterative_start_time)
        tracking_layout.addWidget(self.chk_iterative_tracking)

        self.spin_track_window_hz = QDoubleSpinBox()
        self.spin_track_window_hz.setRange(0.1, 50.0)
        self.spin_track_window_hz.setValue(5.0)
        self.spin_track_window_hz.setSuffix(" Hz")
        self.spin_track_window_hz.setToolTip("Search window width in Hz")
        tracking_layout.addWidget(self.spin_track_window_hz)
        relax_layout.addLayout(tracking_layout)
        # ---------------------------------

        self.chk_log_t2 = QCheckBox("Log T2* View")
        self.chk_log_t2.setToolTip("View Output as Log Scale")
        self.chk_log_t2.stateChanged.connect(self.plot_global_results)
        relax_layout.addWidget(self.chk_log_t2)

        self.btn_batch_run = QPushButton("Run Global Map Analysis")
        self.btn_batch_run.setStyleSheet("background-color: #2196F3; color: white;")
        self.btn_batch_run.setToolTip("Calculate T2* for all detected peaks")
        self.btn_batch_run.clicked.connect(self.run_batch_analysis)
        relax_layout.addWidget(self.btn_batch_run)
        
        self.relax_settings_widget.setLayout(relax_layout)
        ana_tab_layout.addWidget(self.relax_settings_widget)
        
        # Initially hide relax settings (if default is Evolution)
        self.relax_settings_widget.setVisible(False)
        
        ana_tab_layout.addStretch()
        self.tabs_control.addTab(self.tab_analysis, "Analysis")
        
        # --- Bottom Left Status ---
        self.progress_bar = QProgressBar()
        left_layout.addWidget(self.progress_bar)
        
        # --- Right Panel: Visualization ---
        right_splitter = QSplitter(Qt.Vertical)
        
        # 0. Time Domain
        self.fig_time = Figure(figsize=(8, 3))
        self.canvas_time = FigureCanvas(self.fig_time)
        self.ax_time = self.fig_time.add_subplot(111)
        
        time_container = QWidget()
        time_layout = QVBoxLayout(time_container)
        time_layout.addWidget(NavigationToolbar(self.canvas_time, time_container))
        time_layout.addWidget(self.canvas_time)
        right_splitter.addWidget(time_container)
        
        # Top: Spectrum
        self.fig_spec = Figure(figsize=(8, 4))
        self.fig_spec.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.15)
        self.canvas_spec = FigureCanvas(self.fig_spec)
        self.ax_spec = self.fig_spec.add_subplot(111)
        self.canvas_spec.mpl_connect('pick_event', self.on_pick)
        
        spec_container = QWidget()
        spec_layout = QVBoxLayout(spec_container)
        spec_layout.addWidget(NavigationToolbar(self.canvas_spec, spec_container))
        spec_layout.addWidget(self.canvas_spec)
        
        # Spectrum Controls area
        range_group = QGroupBox("View Control")
        range_layout = QHBoxLayout()
        range_layout.setContentsMargins(5, 5, 5, 5)
        range_layout.addWidget(QLabel("Range (Hz):"))
        self.freq_min = QDoubleSpinBox()
        self.freq_min.setRange(0, 5000)
        self.freq_min.setValue(0)
        range_layout.addWidget(self.freq_min)
        range_layout.addWidget(QLabel("-"))
        self.freq_max = QDoubleSpinBox()
        self.freq_max.setRange(0, 5000)
        self.freq_max.setValue(100)
        range_layout.addWidget(self.freq_max)
        btn_update_range = QPushButton("Set")
        btn_update_range.clicked.connect(self.plot_spectrum_traffic_light)
        range_layout.addWidget(btn_update_range)
        range_group.setLayout(range_layout)
        spec_layout.addWidget(range_group)

        # View Mode
        view_mode_group = QGroupBox("Component")
        view_mode_layout = QHBoxLayout()
        view_mode_layout.setContentsMargins(5, 5, 5, 5)
        self.btn_view_mag = QPushButton("Mag")
        self.btn_view_mag.setCheckable(True)
        self.btn_view_mag.setChecked(True)
        self.btn_view_mag.clicked.connect(self.update_view_mode)
        view_mode_layout.addWidget(self.btn_view_mag)
        self.btn_view_real = QPushButton("Real")
        self.btn_view_real.setCheckable(True)
        self.btn_view_real.clicked.connect(self.update_view_mode)
        view_mode_layout.addWidget(self.btn_view_real)
        self.btn_view_imag = QPushButton("Imag")
        self.btn_view_imag.setCheckable(True)
        self.btn_view_imag.clicked.connect(self.update_view_mode)
        view_mode_layout.addWidget(self.btn_view_imag)
        self.view_buttons = [self.btn_view_mag, self.btn_view_real, self.btn_view_imag]
        self.chk_view_abs = QCheckBox("Abs")
        self.chk_view_abs.setChecked(False)
        self.chk_view_abs.stateChanged.connect(self.update_view_mode)
        view_mode_layout.addWidget(self.chk_view_abs)
        view_mode_group.setLayout(view_mode_layout)
        spec_layout.addWidget(view_mode_group)

        right_splitter.addWidget(spec_container)
        
        # Bottom: Evolution/Analysis Plots
        evo_container = QWidget()
        evo_layout = QVBoxLayout(evo_container)
        
        # Only Instruction Label here now
        evo_layout.addWidget(QLabel("Analysis Results:"))
        
        # Plot Area
        self.ana_splitter = QSplitter(Qt.Horizontal)
        
        # Plot 1
        self.plot_container_1 = QWidget()
        l_1 = QVBoxLayout(self.plot_container_1)
        l_1.setContentsMargins(0,0,0,0)
        self.fig_evo = Figure(figsize=(5, 3))
        self.canvas_evo = FigureCanvas(self.fig_evo)
        self.ax_evo = self.fig_evo.add_subplot(111)
        self.canvas_evo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar_evo = NavigationToolbar(self.canvas_evo, self.plot_container_1)
        l_1.addWidget(self.toolbar_evo)
        l_1.addWidget(self.canvas_evo)
        
        # Plot 2
        self.plot_container_2 = QWidget()
        l_2 = QVBoxLayout(self.plot_container_2)
        l_2.setContentsMargins(0,0,0,0)
        self.fig_detail = Figure(figsize=(5, 3))
        self.canvas_detail = FigureCanvas(self.fig_detail)
        self.ax_detail = self.fig_detail.add_subplot(111)
        self.canvas_detail.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar_detail = NavigationToolbar(self.canvas_detail, self.plot_container_2)
        l_2.addWidget(self.toolbar_detail)
        l_2.addWidget(self.canvas_detail)

        self.ana_splitter.addWidget(self.plot_container_1)
        self.ana_splitter.addWidget(self.plot_container_2)
        self.plot_container_2.hide()

        evo_layout.addWidget(self.ana_splitter, stretch=1)
        right_splitter.addWidget(evo_container)
        
        self.main_splitter.addWidget(right_splitter)
        self.main_splitter.setStretchFactor(0, 0) 
        self.main_splitter.setStretchFactor(1, 1)

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

    def save_parameters(self):
        """Save current parameters to JSON file"""
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Parameters", "", "JSON Files (*.json);;All Files (*)"
        )
        if not file_name:
            return
            
        try:
            # 1. Processing Params (from _get_process_params)
            # We construct this manually to include UI states that might not be in _get_process_params yet
            params = {
                'savgol_window': self.savgol_window.value(),
                'savgol_order': self.savgol_order.value(),
                'apod_rate': self.apod_rate.value(),
                'p0': self.p0_slider.value(),
                'p1': self.p1_slider.value(),
                'trunc_start': self.trunc_slider.value(),
                'trunc_end': self.trunc_end_slider.value(),
                'enable_svd': self.chk_svd.isChecked(),
                
                # Peak picking
                'peak_height_abs': self.peak_thr.value(),
                'sys_freq_min': self.freq_min_search.value(),
                'sys_freq_max': self.freq_max_search.value(),
                'peak_window': self.peak_win.value(),
                
                # Noise
                'noise_method': self.combo_noise_method.currentText(),
                'noise_freq_min': self.noise_min.value(),
                'noise_freq_max': self.noise_max.value(),
                'noise_local_window': self.noise_local_win.value(),
                
                # Verdict
                'min_r2': self.thr_r2.value(),
                'min_slope': self.thr_slope.value(),
                
                # Relaxation Analysis Settings
                'relax_unit': self.combo_relax_unit.currentText(),
                'relax_start': self.spin_relax_start.value(),
                'relax_end': self.spin_relax_end.value(),
                'relax_points': self.spin_relax_points.value(),
                'relax_zerofill': self.chk_relax_zerofill.isChecked(),
                'relax_use_tracking': self.chk_iterative_tracking.isChecked(),
                'relax_track_hz': self.spin_track_window_hz.value(),
                
                # Plot Settings
                'view_freq_min': self.freq_min.value(),
                'view_freq_max': self.freq_max.value(),
                'analysis_mode': self.combo_analysis_mode.currentText()
            }
            
            with open(file_name, 'w') as f:
                json.dump(params, f, indent=4)
                
            self.statusBar().showMessage(f"Parameters saved to {file_name}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save parameters:\n{str(e)}")

    def load_parameters(self):
        """Load parameters from JSON file"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Load Parameters", "", "JSON Files (*.json);;All Files (*)"
        )
        if not file_name:
            return
            
        try:
            with open(file_name, 'r') as f:
                params = json.load(f)
            
            # Helper to safely set values
            def set_val(widget, key, cast=float):
                if key in params:
                    # Handle SliderSpinBox (has spinbox attribute) vs QDoubleSpinBox/QSpinBox
                    val = cast(params[key])
                    if hasattr(widget, 'spinbox'): 
                        widget.spinbox.setValue(val)
                    else:
                        widget.setValue(val)

            # Block signals to avoid massive processing triggers
            self.blockSignals(True)
            
            set_val(self.savgol_window, 'savgol_window', int)
            set_val(self.savgol_order, 'savgol_order', int)
            set_val(self.apod_rate, 'apod_rate', float)
            set_val(self.p0_slider, 'p0', float)
            set_val(self.p1_slider, 'p1', float)
            set_val(self.trunc_slider, 'trunc_start', int)
            set_val(self.trunc_end_slider, 'trunc_end', int)
            
            if 'enable_svd' in params:
                self.chk_svd.setChecked(params['enable_svd'])
                
            set_val(self.peak_thr, 'peak_height_abs', float)
            set_val(self.freq_min_search, 'sys_freq_min', float)
            set_val(self.freq_max_search, 'sys_freq_max', float)
            set_val(self.peak_win, 'peak_window', int)
            
            if 'noise_method' in params:
                self.combo_noise_method.setCurrentText(params['noise_method'])
            
            set_val(self.noise_min, 'noise_freq_min', float)
            set_val(self.noise_max, 'noise_freq_max', float)
            set_val(self.noise_local_win, 'noise_local_window', int)
            
            set_val(self.thr_r2, 'min_r2', float)
            set_val(self.thr_slope, 'min_slope', float)
            
            if 'relax_unit' in params:
                self.combo_relax_unit.setCurrentText(params['relax_unit'])
            set_val(self.spin_relax_start, 'relax_start', float)
            set_val(self.spin_relax_end, 'relax_end', float)
            set_val(self.spin_relax_points, 'relax_points', int)
            if 'relax_zerofill' in params:
                self.chk_relax_zerofill.setChecked(params['relax_zerofill'])
            if 'relax_use_tracking' in params:
                self.chk_iterative_tracking.setChecked(params['relax_use_tracking'])
            if 'relax_track_hz' in params:
                self.spin_track_window_hz.setValue(params['relax_track_hz'])
            
            if 'view_freq_min' in params: self.freq_min.setValue(params['view_freq_min'])
            if 'view_freq_max' in params: self.freq_max.setValue(params['view_freq_max'])
            
            if 'analysis_mode' in params:
                self.combo_analysis_mode.setCurrentText(params['analysis_mode'])

            self.blockSignals(False)
            self.statusBar().showMessage(f"Parameters loaded from {file_name}")
            
            # Trigger updates
            self.plot_spectrum_traffic_light()
            self.update_verdicts()
            
        except Exception as e:
            self.blockSignals(False)
            QMessageBox.critical(self, "Error", f"Failed to load parameters:\n{str(e)}")

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
            'peak_height_abs': float(self.peak_thr.value()),
            'peak_window': int(self.peak_win.value()),
            'search_freq_min': float(self.freq_min_search.value()),
            'search_freq_max': float(self.freq_max_search.value()),
            'noise_freq_min': float(self.noise_min.value()),
            'noise_freq_max': float(self.noise_max.value()),
            'noise_method': 'global' if self.combo_noise_method.currentText() == "Global Region" else 'local',
            'local_noise_window': int(self.noise_local_win.value()),
            'detect_mode': detect_mode # Pass to worker -> validator
        }

    def request_processing_update(self):
        if self.raw_avg_data is not None:
            self.update_timer.start()

    def update_relax_ui_state(self, unit):
        if unit == "Points":
             self.spin_relax_start.setSuffix(" pts")
             self.spin_relax_start.setDecimals(0)
             self.spin_relax_end.setSuffix(" pts")
             self.spin_relax_end.setDecimals(0)
        else:
             self.spin_relax_start.setSuffix(" ms")
             self.spin_relax_start.setDecimals(1)
             self.spin_relax_end.setSuffix(" ms")
             self.spin_relax_end.setDecimals(1)

    def sync_iterative_start_time(self):
        """Called when Iterative Track is toggled. Enforce start time = trunc start."""
        if not self.chk_iterative_tracking.isChecked():
            # Re-enable if manual control desired (optional, maybe keep enabled if user wants to change?)
            # But if we enforced lock, we might want to let them change it back.
            self.spin_relax_start.setEnabled(True)
            return

        # Lock to Truncation Start
        trunc_pts = self.trunc_slider.value()
        
        # Disable to indicate lock? Or just set value?
        # User requested "lock", so disabling is clearer visual feedback.
        # But maybe they want to see the value.
        self.spin_relax_start.setEnabled(False) # Visual lock
        
        unit = self.combo_relax_unit.currentText()
        
        self.blockSignals(True)
        if unit == "Points":
            self.spin_relax_start.setValue(trunc_pts)
        else:
             # Convert using sampling rate
             if self.sampling_rate and self.sampling_rate > 0:
                 t_ms = (trunc_pts / self.sampling_rate) * 1000.0
                 self.spin_relax_start.setValue(t_ms)
             else:
                 # Fallback if no data loaded yet
                 pass
        self.blockSignals(False)
        self.statusBar().showMessage(f"Iterative Mode: Start Time locked to {trunc_pts} pts")

    def on_analysis_mode_changed(self, index):
        mode_text = self.combo_analysis_mode.currentText()
        is_global = mode_text.startswith("Global")
        is_relax = mode_text.startswith("Relaxation") or is_global

        # 1. Manage Settings Visibility
        if hasattr(self, 'relax_settings_widget'):
            self.relax_settings_widget.setVisible(is_relax)
            
        # 2. Manage Batch Button Visibility
        if hasattr(self, 'btn_batch_run'):
            self.btn_batch_run.setVisible(is_global)

        # 3. Manage Detail Plot Visibility
        if hasattr(self, 'plot_container_2'):
            self.plot_container_2.setVisible(is_global)

        # 4. Reset Plots
        self.fig_evo.clear()
        self.ax_evo = self.fig_evo.add_subplot(111)
        
        if is_global:
            self.ax_evo.text(0.5, 0.5, "Click 'Run Global Map' to start", ha='center', va='center')
            self.ax_detail.clear()
            self.ax_detail.grid(True)
            self.canvas_detail.draw()
        else:
            self.ax_evo.text(0.5, 0.5, "Click a peak to analyze", ha='center', va='center')
        
        self.canvas_evo.draw()

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
        self.sampling_rate = sampling_rate # Store for general use
        
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
        
        # update dynamic ranges
        try:
             # Amplitude
             max_amp = np.max(np.abs(spec))
             # If current max range is significantly smaller than actual max, update it
             # Use 1.2x headroom
             self.peak_thr.set_range(0, max_amp * 1.5)
             
             # Frequency
             max_freq = np.max(freqs)
             if max_freq > 0:
                 self.freq_min_search.set_range(0, max_freq)
                 self.freq_max_search.set_range(0, max_freq)
                 self.noise_min.set_range(0, max_freq)
                 self.noise_max.set_range(0, max_freq)

        except Exception as e:
            print(f"Error updating dynamic ranges: {e}")

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

        try:
            self.fig_time.tight_layout()
        except:
            pass
        self.canvas_time.draw()

    def start_analysis(self):
        if not self.folder_paths:
            QMessageBox.warning(self, "No Data", "Please add at least one folder.")
            return

        params = self._get_process_params()
        
        self.btn_run.setEnabled(False)
        self.btn_reprocess.setEnabled(False)
        self.btn_load.setEnabled(False)
        self.progress_bar.setRange(0, 0) # Indeterminate
        
        self.worker = ValidationWorker(self.folder_paths, params)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.error.connect(self.on_error)
        self.worker.progress.connect(self.update_status)
        self.worker.start()
        
    def update_status(self, msg):
        self.statusBar().showMessage(msg)
        
    def on_error(self, msg):
        self.btn_run.setEnabled(True)
        self.btn_reprocess.setEnabled(True)
        self.btn_load.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.statusBar().showMessage("Error")
        QMessageBox.critical(self, "Analysis Failed", msg)
        
    def on_analysis_finished(self, results, freqs, spec, evo_data):
        self.btn_run.setEnabled(True)
        self.btn_reprocess.setEnabled(True)
        self.btn_load.setEnabled(True)
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
        self.plot_spectrum_traffic_light(preserve_view=True)
        
    def update_noise_ui_visibility(self, method):
        is_global = (method == "Global Region")
        self.noise_global_group.setVisible(is_global)
        self.noise_local_group.setVisible(not is_global)
        self.plot_spectrum_traffic_light(preserve_view=True)

    def update_view_mode(self):
        sender = self.sender()
        if not sender: return
        
        # Enforce exclusivity
        for btn in self.view_buttons:
            if btn != sender:
                btn.setChecked(False)
        sender.setChecked(True)
        
        self.plot_spectrum_traffic_light() # Changing mode (Real/Imag) naturally requires re-scale

    def plot_spectrum_traffic_light(self, preserve_view=False):
        # Capture current Y-lim before clearing if preserving view
        current_ylim = self.ax_spec.get_ylim() if preserve_view else None
        
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
        
        # Guard against empty or singular data (prevents Singular Matrix error in axvspan)
        if len(pos_freqs) < 2 or np.ptp(pos_freqs) == 0:
            self.canvas_spec.draw()
            return

        self.ax_spec.plot(pos_freqs, pos_data, 'k-', linewidth=0.8, alpha=0.6, label=f'Spectrum ({mode_label})')
        
        # --- Visualization of Detection Params ---
        # Wrap decorations in try-except to handle singular matrix errors during layout transitions
        try:
            # 1. Amplitude Threshold
            thr_val = self.peak_thr.value()
            self.ax_spec.axhline(y=thr_val, color='#FFA500', linestyle='--', linewidth=1, alpha=0.7, label='Height Threshold')
            
            # 2. Search Window
            f_min = self.freq_min_search.value()
            f_max = self.freq_max_search.value()
            
            # Highlight the VALID search region (or grey out the ignored)
            # Greying out outside is clearer
            # Left side
            if f_min > 0:
                self.ax_spec.axvspan(0, f_min, color='gray', alpha=0.1)
            # Right side
            current_max_freq = np.max(pos_freqs)
            if f_max < current_max_freq:
                self.ax_spec.axvspan(f_max, current_max_freq, color='gray', alpha=0.1, label='Ignored Region')
            
            # Add vertical lines for bounds
            self.ax_spec.axvline(x=f_min, color='gray', linestyle=':', alpha=0.5)
            self.ax_spec.axvline(x=f_max, color='gray', linestyle=':', alpha=0.5)

            # 3. Noise Region
            if self.combo_noise_method.currentText() == "Global Region":
                n_min = self.noise_min.value()
                n_max = self.noise_max.value()
                # Red semi-transparent area for noise
                self.ax_spec.axvspan(n_min, n_max, color='red', alpha=0.05, label='Noise Calc Region')
                self.ax_spec.text((n_min+n_max)/2, thr_val*1.1, "Noise", color='red', fontsize=8, ha='center')
        except Exception as e:
            print(f"Warning: Could not draw spectrum decorations: {e}")

        # Plot Candidates (Overlaid on the selected view)

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
        self.ax_spec.set_xlim(left=x_min, right=x_max)
        
        # Restore Y-Limits if requested, otherwise Auto-scale
        if preserve_view and current_ylim is not None:
            self.ax_spec.set_ylim(current_ylim)
        else:
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
        
        # NOTE: Do NOT call tight_layout here repeatedly. 
        # It causes figure height jitter/collapse when updating frequently.
        # self.fig_spec.tight_layout() 
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
            if self.combo_analysis_mode.currentText().startswith("Relaxation"):
                self.run_relaxation_analysis(closest_row['Freq_Hz'])
            else:
                self.plot_evolution(closest_row['Index'])

    def run_relaxation_analysis(self, target_freq):
        if self.raw_avg_data is None:
            return
            
        self.ax_evo.clear()
        self.ax_evo.text(0.5, 0.5, "Calculating Relaxation...", ha='center', va='center')
        self.canvas_evo.draw()
        
        params = self._get_process_params()
        # Use RAW average data for this analysis to be accurate
        # (Assuming raw_avg_data is the time domain signal)
        
        sampling_rate = self.sampling_rate
        
        # Get Time Range from UI
        val_start = self.spin_relax_start.value()
        val_end = self.spin_relax_end.value()
        unit = self.combo_relax_unit.currentText()
        
        if unit == "Points":
             t_start = val_start / sampling_rate
             t_end = val_end / sampling_rate
        else:
             t_start = val_start / 1000.0
             t_end = val_end / 1000.0
             
        points = self.spin_relax_points.value()
        zero_fill_front = self.chk_relax_zerofill.isChecked()
        
        self.relax_worker = RelaxationWorker(
            self.raw_avg_data, 
            target_freq, 
            sampling_rate,
            params,
            t_start=t_start,
            t_end=t_end,
            points=points,
            zero_fill_front=zero_fill_front
        )
        # Need sampling rate. self.loader_worker might be gone or recycled.
        # But we stored sampling_rate in on_loading_finished?
        # Let's check on_loading_finished
        
        self.relax_worker.finished.connect(self.plot_relaxation_results)
        self.relax_worker.progress.connect(self.on_worker_progress)
        self.relax_worker.error.connect(lambda e: QMessageBox.warning(self, "Analysis Error", e))
        self.relax_worker.start()

    def on_worker_progress(self, msg, val):
        self.statusBar().showMessage(msg)
        self.progress_bar.setValue(val)

    def plot_relaxation_results(self, times, amps, fit_x, fit_y, t2, r2):
        self.progress_bar.setValue(100)
        self.statusBar().showMessage("Relaxation Analysis Complete")
        self.ax_evo.clear()
        
        # Plot Data
        self.ax_evo.scatter(times, amps, c='b', alpha=0.6, label='Truncation Amp')
        
        # Plot Fit
        if len(fit_x) > 0:
            self.ax_evo.plot(fit_x, fit_y, 'r-', linewidth=2, label=f'Fit: T2*={t2*1000:.1f}ms')
            
        self.ax_evo.set_xlabel("Truncation Start Time (s)")
        self.ax_evo.set_ylabel("Peak Amplitude")
        self.ax_evo.set_title(f"Relaxation Analysis (R2={r2:.3f})")
        self.ax_evo.legend()
        
        try:
             self.fig_evo.tight_layout()
        except:
             pass
        self.canvas_evo.draw()

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
        
        try:
             self.fig_evo.tight_layout()
        except:
             pass
        self.canvas_evo.draw()


    def run_batch_analysis(self):
        if self.current_results is None or self.raw_avg_data is None:
             QMessageBox.warning(self, "No Data", "Please process data first.")
             return
             
        # Filter valid peaks
        valid_peaks = self.current_results[self.current_results['Verdict'] == True]
        if valid_peaks.empty:
            QMessageBox.information(self, "Info", "No valid peaks detected.")
            return
            
        # Params
        params = self._get_process_params()
        sampling_rate = self.sampling_rate
        
        val_start = self.spin_relax_start.value()
        val_end = self.spin_relax_end.value()
        unit = self.combo_relax_unit.currentText()
        
        if unit == "Points":
             t_start = val_start / sampling_rate
             t_end = val_end / sampling_rate
        else:
             t_start = val_start / 1000.0
             t_end = val_end / 1000.0
             
        points = self.spin_relax_points.value()
        zero_fill_front = self.chk_relax_zerofill.isChecked()
        
        # Iterative Tracking Settings
        use_tracking = self.chk_iterative_tracking.isChecked()
        track_win_hz = self.spin_track_window_hz.value()

        # [Iterative Mode Correction]
        # If tracking is enabled, the starting point MUST match the point where peaks were detected (Truncation Start).
        # Otherwise, the initial guess (Global Peaks) might be invalid for a different time point due to drift.
        if use_tracking:
            trunc_pts = self.trunc_slider.value()
            t_trunc = trunc_pts / sampling_rate
            
            # Enforce synchronization
            if abs(t_start - t_trunc) > 1e-6: # Float comparison
                print(f"Iterative Mode: Auto-locking Start Time to Truncation Start ({t_trunc*1000:.1f} ms)")
                self.statusBar().showMessage(f"Iterative Mode: Locked Start Time to {t_trunc*1000:.1f} ms (Truncation Point)")
                
                t_start = t_trunc
                # Sync UI
                self.blockSignals(True)
                if unit == "Points":
                    self.spin_relax_start.setValue(trunc_pts)
                else:
                     self.spin_relax_start.setValue(t_trunc * 1000.0)
                self.blockSignals(False)

        # UI State
        self.btn_batch_run.setEnabled(False)
        self.progress_bar.setRange(0, 100) # Ensure range is set
        self.progress_bar.setValue(0)
        self.ax_evo.clear()
        self.ax_evo.text(0.5, 0.5, "Running Batch Analysis...", ha='center', va='center')
        self.canvas_evo.draw()
        
        # Worker
        self.batch_worker = BatchRelaxationWorker(
            self.raw_avg_data,
            valid_peaks, 
            sampling_rate,
            params,
            t_start, t_end, points,
            zero_fill_front,
            use_tracking,
            track_win_hz
        )
        self.batch_worker.finished.connect(self.on_batch_analysis_finished)
        self.batch_worker.progress.connect(self.on_worker_progress)
        self.batch_worker.error.connect(lambda e: [self.btn_batch_run.setEnabled(True), QMessageBox.critical(self, "Batch Error", f"Analysis failed: {e}")])
        self.batch_worker.start()

    def on_batch_analysis_finished(self, results):
        self.btn_batch_run.setEnabled(True)
        self.progress_bar.setValue(100)
        self.statusBar().showMessage("Batch Analysis Complete")
        
        self.batch_results_summary = results['summary']
        self.batch_results_details = results['details']
        
        self.plot_global_results()
        
    def plot_global_results(self):
        # Reset figure layout to prevent colorbar shrinking
        self.fig_evo.clear()
        self.ax_evo = self.fig_evo.add_subplot(111)
        self.cbar_evo = None # Reset ref
        
        self.ax_detail.clear() # Clear detail too
        
        if self.batch_results_summary is None or self.batch_results_summary.empty:
             self.ax_evo.text(0.5, 0.5, "No results or Global Analysis not run", ha='center')
             self.canvas_evo.draw()
             return

        df = self.batch_results_summary
             
        # Scatter Freq vs T2*
        # Use simple scatter with picker=5
        sc = self.ax_evo.scatter(df['Freq_Hz'], df['T2_star_ms'], c=df['R2'], cmap='viridis', picker=5, s=40, edgecolors='k')
        self.ax_evo.set_xlabel('Frequency (Hz)')
        self.ax_evo.set_ylabel('T2* (ms)')
        self.ax_evo.set_title(f"Global T2* Map (n={len(df)})")
        self.ax_evo.grid(True, alpha=0.3)
        
        # Log Scale Option
        if hasattr(self, 'chk_log_t2') and self.chk_log_t2.isChecked():
            self.ax_evo.set_yscale('log')

        # Colorbar - use constrained layout or just add it to refreshed figure
        self.cbar_evo = self.fig_evo.colorbar(sc, ax=self.ax_evo, label='Fit R2')
        
        self.canvas_evo.draw()
        
        # Connect pick event for this canvas if not connected
        if not hasattr(self, '_global_pick_cid'):
             self._global_pick_cid = self.canvas_evo.mpl_connect('pick_event', self.on_global_pick)

    def on_global_pick(self, event):
        # Only act if we are in Global mode
        if not self.combo_analysis_mode.currentText().startswith("Global"):
            return

        ind = event.ind[0]
        df = self.batch_results_summary
        if ind < len(df):
            row = df.iloc[ind]
            freq = row['Freq_Hz']
            self.plot_detail_curve(freq)
            
    def plot_detail_curve(self, freq):
        # Retrieve detail data
        data = self.batch_results_details.get(freq)
        if not data: return
        
        self.ax_detail.clear()
        
        times = data['times'] * 1000 # to ms
        amps = data['amps']
        
        self.ax_detail.plot(times, amps, 'bo', label='Data')
        
        if len(data['fit_x']) > 0:
             fit_x_ms = data['fit_x'] * 1000
             fit_y = data['fit_y']
             t2 = data['t2']
             self.ax_detail.plot(fit_x_ms, fit_y, 'r-', label=f'Fit (T2*={t2*1000:.1f}ms)')
        
        self.ax_detail.set_xlabel('Delay (ms)')
        self.ax_detail.set_ylabel('Amplitude')
        self.ax_detail.set_title(f"Decay @ {freq:.1f} Hz")
        self.ax_detail.legend(fontsize='small')
        self.ax_detail.grid(True)
        
        self.canvas_detail.draw()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
