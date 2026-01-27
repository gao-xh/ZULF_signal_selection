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
from src.processing import Processor, CurveFitter
from src.validator import SignalValidator
from src.auto_phase import auto_phase_entropy, apply_phase_correction
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

    def __init__(self, full_data, target_freq, sampling_rate, params, t_start=0, t_end=0, points=50, zero_fill_front=False, measure_mode="Magnitude"):
        super().__init__()
        self.data = full_data
        self.target_freq = target_freq
        self.fs = sampling_rate
        self.params = params
        self.t_start = t_start
        self.t_end = t_end
        self.points = points
        self.zero_fill_front = zero_fill_front
        self.measure_mode = measure_mode

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
            # Pre-retrieve global phase params for Fixed mode
            global_p0 = self.params.get('p0', 0)
            global_p1 = self.params.get('p1', 0)

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
                    padded[:] = 0
                    padded[start_iter:] = processed_segment
                else:
                    padded[:len(processed_segment)] = processed_segment
                    padded[len(processed_segment):] = 0
                
                # 3. FFT
                spec = scipy.fft.fft(padded)
                
                # 4. Measure Mode Handling
                if self.measure_mode == "Real (Auto-Phased)":
                    # Run auto-phase on this specific slice
                    # Note: This is computationally expensive!
                    p0, p1 = auto_phase_entropy(spec)
                    spec_phased = apply_phase_correction(spec, p0, p1)
                    measure_data = np.real(spec_phased)
                    
                elif self.measure_mode == "Real (Fixed Phase)":
                    # Use global params
                    spec_phased = apply_phase_correction(spec, global_p0, global_p1)
                    measure_data = np.real(spec_phased)
                    
                else: # "Magnitude"
                    measure_data = np.abs(spec)
                
                # 5. Measure Peak
                idx_start = max(0, target_idx - search_r)
                idx_end = min(N, target_idx + search_r + 1)
                
                if idx_end > idx_start:
                    # For Real parts, we might care about signed max or just max value
                    # Usually peak height assumes positive peak after phasing.
                    peak_amp = np.max(measure_data[idx_start:idx_end])
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

    def __init__(self, full_data, target_freqs_df, sampling_rate, params, t_start=0, t_end=0, points=50, zero_fill_front=False, use_tracking=False, track_win_hz=5.0, noise_threshold=None, measure_mode="Magnitude"):
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
        self.noise_threshold = noise_threshold
        self.measure_mode = measure_mode
        self.zero_fill_front = zero_fill_front
        self.use_tracking = use_tracking
        self.track_win_hz = track_win_hz
        self.noise_threshold = noise_threshold

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
            # Dictionary: freq -> {times:[], amps:[], freqs:[]}
            curve_data = { row['Freq_Hz']: {'times': [], 'amps': [], 'freqs': []} for _, row in self.target_freqs.iterrows() }
            
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
# Retrieve global phase params
            global_p0 = self.params.get('p0', 0)
            global_p1 = self.params.get('p1', 0)

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
                 
                 # 3. Measure Mode Handling
                 if self.measure_mode == "Real (Auto-Phased)":
                    p0, p1 = auto_phase_entropy(spec)
                    spec_phased = apply_phase_correction(spec, p0, p1)
                    measure_data = np.real(spec_phased)
                 
                 elif self.measure_mode == "Real (Fixed Phase)":
                    spec_phased = apply_phase_correction(spec, global_p0, global_p1)
                    measure_data = np.real(spec_phased)
                    
                 else: # "Magnitude"
                    measure_data = np.abs(spec)
                 
                 current_time = start_iter / self.fs
                 
                 # Prepare next iteration updates
                 next_indices_update = {}

                 # 4. Measure All Peaks
                 for f_key in curve_data.keys():
                     # Use current tracked index
                     idx_center = current_peak_indices[f_key]
                     
                     idx_start = max(0, idx_center - search_r)
                     idx_end = min(N, idx_center + search_r + 1)
                     
                     if idx_end > idx_start:
                        segment_view = measure_data[idx_start:idx_end]
                        peak_amp = np.max(segment_view)
                        
                        # Logic: If tracking, find local max index to update center
                        current_peak_val_hz = f_key
                        if self.use_tracking:
                            local_max = np.argmax(segment_view)
                            abs_max_idx = idx_start + local_max
                            
                            # DRIFT GUARD: Prevent tracking from walking too far from initial frequency
                            # This fixes the "messy tail" artifact where tracker chases noise.
                            initial_idx = freq_to_idx[f_key]
                            max_drift_bins = 20 # Limit drift to ~20 bins (configurable?)
                            
                            if abs(abs_max_idx - initial_idx) > max_drift_bins:
                                 # If drifted too far, snap back to initial or current center?
                                 # Snap to initial is safer for T2* decay (signals shouldn't shift 100s of Hz)
                                 abs_max_idx = initial_idx 
                            
                            next_indices_update[f_key] = abs_max_idx
                            current_peak_val_hz = abs_max_idx * hz_per_point
                        else:
                            # If not tracking, we take the max in window, so freq implies the max index
                            local_max = np.argmax(segment_view)
                            abs_max_idx = idx_start + local_max
                            current_peak_val_hz = abs_max_idx * hz_per_point

                     else:
                        peak_amp = 0
                        current_peak_val_hz = f_key
                        if self.use_tracking:
                            next_indices_update[f_key] = idx_center # Keep if lost
                     
                     curve_data[f_key]['times'].append(current_time)
                     curve_data[f_key]['amps'].append(peak_amp)
                     curve_data[f_key]['freqs'].append(current_peak_val_hz)
                
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
                 # 1. Standard Fit (Raw)
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
                 
                 # 2. Filtered Fit (Remove Oscillation) - OPTIONAL but computed for comparison
                 # We keep it lightweight if possible, but user wants it global.
                 # Using CurveFitter from processing
                 t2_filt = 0
                 r2_filt = 0
                 filt_details = {}  # Store filtered curve data
                 
                 try:
                     # Only run if enough points
                     if len(times) > 10:
                         # Pass noise threshold to constrain offset C
                         filt_res = CurveFitter.remove_oscillation_fft(times, amps, noise_level=self.noise_threshold)
                         # Store basic filtered signal if available
                         if filt_res['status'] == 'success':
                             filt_details['times_filt'] = filt_res.get('times', times)
                             filt_details['amps_filt'] = filt_res.get('amps_filtered', []) 
                         
                         if filt_res['status'] == 'success' and 'fit_result' in filt_res:
                            fr = filt_res['fit_result']
                            if fr['status'] == 'success':
                                t2_filt = fr['T2']
                                r2_filt = fr['R2']
                                filt_details['fit_x_filt'] = fr['fit_x']
                                filt_details['fit_y_filt'] = fr['fit_y']
                 except: 
                     pass
                 
                 summary_list.append({
                     'Freq_Hz': f_key,
                     'T2_star_ms': t2 * 1000.0,
                     'R2': r2,
                     'T2_star_filt_ms': t2_filt * 1000.0,
                     'R2_filt': r2_filt
                 })
                 
                 # Combine standard details with optional filtered details
                 detail_entry = {
                     'times': times,
                     'amps': amps,
                     'freqs': np.array(data_c['freqs']),
                     'fit_x': fit_x,
                     'fit_y': fit_y,
                     't2': t2,
                     'r2': r2
                 }
                 detail_entry.update(filt_details)
                 detailed_results[f_key] = detail_entry
                 
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
        
        # Auto Phase Button
        self.btn_auto_phase = QPushButton("Auto Phase (Entropy)")
        self.btn_auto_phase.setToolTip("Automatically correct phase using Minimum Entropy algorithm")
        self.btn_auto_phase.clicked.connect(self.run_auto_phase)
        proc_layout.addWidget(self.btn_auto_phase)
        
        # View Mode / Data Mode (Global)
        self.view_mode_group = QGroupBox("Processing Target (Mode)")
        view_mode_layout = QHBoxLayout()
        view_mode_layout.setContentsMargins(2, 2, 2, 2)
        
        self.btn_view_mag = QPushButton("Mag")
        self.btn_view_mag.setCheckable(True)
        self.btn_view_mag.setChecked(True)
        self.btn_view_mag.clicked.connect(self.update_view_mode)
        
        self.btn_view_real = QPushButton("Real")
        self.btn_view_real.setCheckable(True)
        self.btn_view_real.clicked.connect(self.update_view_mode)
        
        self.btn_view_imag = QPushButton("Imag")
        self.btn_view_imag.setCheckable(True)
        self.btn_view_imag.clicked.connect(self.update_view_mode)
        
        view_mode_layout.addWidget(self.btn_view_mag)
        view_mode_layout.addWidget(self.btn_view_real)
        view_mode_layout.addWidget(self.btn_view_imag)
        
        self.view_buttons = [self.btn_view_mag, self.btn_view_real, self.btn_view_imag]
        
        self.chk_view_abs = QCheckBox("Abs")
        self.chk_view_abs.setChecked(False)
        self.chk_view_abs.stateChanged.connect(self.update_view_mode)
        view_mode_layout.addWidget(self.chk_view_abs)
        
        self.view_mode_group.setLayout(view_mode_layout)
        proc_layout.addWidget(self.view_mode_group)
        
        proc_group.setLayout(proc_layout)
        proc_tab_layout.addWidget(proc_group)
        
        # Bottom Button
        self.btn_reprocess = QPushButton("Refresh Processing")
        self.btn_reprocess.clicked.connect(self.run_processing)
        proc_tab_layout.addWidget(self.btn_reprocess)
        
        # --- Spectrogram Controls ---
        spec_group = QGroupBox("Spectrogram")
        spec_layout_box = QVBoxLayout()
        spec_layout_box.setContentsMargins(2, 2, 2, 2)
        
        # Button
        self.btn_show_spectrogram = QPushButton("Show Spectrogram")
        self.btn_show_spectrogram.clicked.connect(self.update_spectrogram)
        spec_layout_box.addWidget(self.btn_show_spectrogram)
        
        # Params Row
        row_spec = QHBoxLayout()
        row_spec.addWidget(QLabel("Win:"))
        self.combo_spec_window = QComboBox()
        # Options: (Label, NFFT)
        self.combo_spec_window.addItem("High Time Res (256)", 256)
        self.combo_spec_window.addItem("Balanced (1024)", 1024)
        self.combo_spec_window.addItem("High Freq Res (4096)", 4096)
        self.combo_spec_window.setCurrentIndex(1) # Default Balanced
        row_spec.addWidget(self.combo_spec_window)
        
        self.chk_spec_abs = QCheckBox("Abs(Freq)")
        self.chk_spec_abs.setToolTip("Take absolute value of frequency axis (Mirror negative freqs to positive)")
        self.chk_spec_abs.setChecked(True)
        row_spec.addWidget(self.chk_spec_abs)

        self.chk_spec_log = QCheckBox("Log(dB)")
        self.chk_spec_log.setChecked(True)
        row_spec.addWidget(self.chk_spec_log)
        
        spec_layout_box.addLayout(row_spec)
        spec_group.setLayout(spec_layout_box)
        proc_tab_layout.addWidget(spec_group)
        # ----------------------------

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
        self.combo_analysis_mode.addItems(["Signal Evolution (SNR vs N)", "Dephasing Analysis (T2*)", "Global T2* Map"])
        self.combo_analysis_mode.currentIndexChanged.connect(self.on_analysis_mode_changed)
        mode_l.addWidget(self.lbl_analysis_mode)
        mode_l.addWidget(self.combo_analysis_mode)
        mode_group.setLayout(mode_l)
        ana_tab_layout.addWidget(mode_group)

        # Relaxation Params (Container)
        self.relax_settings_widget = QGroupBox("Dephasing Parameters (T2*)")
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
        # Auto-Phase Logic for Analysis (New)
        self.chk_auto_phase_step = QCheckBox("Auto-Phase Each Step")
        self.chk_auto_phase_step.setToolTip("Run Minimum Entropy Phasing for EVERY time slice before measurement.\nRequires 'Real' mode selected in Processing tab.")
        self.chk_auto_phase_step.stateChanged.connect(self.request_analysis_update)
        relax_layout.addWidget(self.chk_auto_phase_step)

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
        
        # Display Mode Selection (Raw / Filtered / Overlay)
        self.combo_t2_display_mode = QComboBox()
        self.combo_t2_display_mode.addItems(["Show Raw T2*", "Show Filtered T2*", "Show Overlay"])
        self.combo_t2_display_mode.setToolTip("Select T2* visualization mode")
        self.combo_t2_display_mode.currentIndexChanged.connect(self.plot_global_results)
        relax_layout.addWidget(self.combo_t2_display_mode)

        self.btn_batch_run = QPushButton("Run Global Map Analysis")
        self.btn_batch_run.setStyleSheet("background-color: #2196F3; color: white;")
        self.btn_batch_run.setToolTip("Calculate T2* for all detected peaks")
        self.btn_batch_run.clicked.connect(self.run_batch_analysis)
        relax_layout.addWidget(self.btn_batch_run)
        
        self.relax_settings_widget.setLayout(relax_layout)
        ana_tab_layout.addWidget(self.relax_settings_widget)
        
        # --- Advanced Analysis Group (New) ---
        self.adv_analysis_group = QGroupBox("Advanced Decay Analysis")
        adv_layout = QVBoxLayout()
        
        hbox_adv = QHBoxLayout()
        self.btn_fit_envelope = QPushButton("Fit Peak Envelope")
        self.btn_fit_envelope.setToolTip("Fit T2* using only localized peaks (ignoring beat valleys)")
        self.btn_fit_envelope.clicked.connect(self.run_envelope_fit)
        
        self.btn_fit_cosine = QPushButton("Fit J-Coupling")
        self.btn_fit_cosine.setToolTip("Fit Damped Cosine to extract Beat Frequency (J)")
        self.btn_fit_cosine.clicked.connect(self.run_cosine_fit)
        
        self.btn_remove_osc = QPushButton("Remove Oscillation (FFT)")
        self.btn_remove_osc.setToolTip("Remove aliased carrier oscillation using FFT Notch Filter")
        self.btn_remove_osc.clicked.connect(self.run_oscillation_filter)
        
        hbox_adv.addWidget(self.btn_fit_envelope)
        hbox_adv.addWidget(self.btn_fit_cosine)
        hbox_adv.addWidget(self.btn_remove_osc)
        adv_layout.addLayout(hbox_adv)
        
        self.lbl_adv_result = QLabel("Result: (Select a point in the Global Map)")
        self.lbl_adv_result.setStyleSheet("color: #333; font-weight: bold;")
        self.lbl_adv_result.setWordWrap(True)
        adv_layout.addWidget(self.lbl_adv_result)
        
        self.adv_analysis_group.setLayout(adv_layout)
        ana_tab_layout.addWidget(self.adv_analysis_group)
        self.adv_analysis_group.setVisible(False) # Hidden by default
        # -------------------------------------
        
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

        # View Mode - Removed from here
        # view_mode_group = QGroupBox("Component")
        # ... (Moved to Processing Tab)

        right_splitter.addWidget(spec_container)
        
        # Bottom: Combined Analysis Area (Spectrogram + T2* Analysis)
        # User requested side-by-side layout: Spectrogram Left, T2* Right
        self.analysis_splitter = QSplitter(Qt.Horizontal)

        # --- Panel 1: Spectrogram (Left) ---
        self.panel_spectrogram = QWidget()
        spec_tab_layout = QVBoxLayout(self.panel_spectrogram)
        spec_tab_layout.setContentsMargins(0,0,0,0)
        
        self.fig_stft = Figure(figsize=(5, 3))
        self.canvas_stft = FigureCanvas(self.fig_stft)
        self.ax_stft = self.fig_stft.add_subplot(111)
        self.canvas_stft.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.toolbar_stft = NavigationToolbar(self.canvas_stft, self.panel_spectrogram)
        
        # Connect Click Event (T2* Analysis from Spectrogram)
        self.canvas_stft.mpl_connect('button_press_event', self.on_spectrogram_click)
        
        spec_tab_layout.addWidget(self.toolbar_stft)
        spec_tab_layout.addWidget(self.canvas_stft)
        
        self.analysis_splitter.addWidget(self.panel_spectrogram)

        # --- Panel 2: T2* Analysis (Right) ---
        self.panel_t2 = QWidget()
        ana_tab_layout = QVBoxLayout(self.panel_t2)
        ana_tab_layout.setContentsMargins(0,0,0,0)
        
        self.ana_splitter = QSplitter(Qt.Vertical) # Changed to Vertical internal split for T2 plots
        
        # Plot 1 (T2* Map / Evolution)
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
        
        # Plot 2 (Detail Curve - Hidden by default or secondary)
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
        
        ana_tab_layout.addWidget(self.ana_splitter)
        
        self.analysis_splitter.addWidget(self.panel_t2)
        
        # Set Ratios: Spectrogram takes priority (e.g. 60%), T2 takes 40%
        self.analysis_splitter.setStretchFactor(0, 3)
        self.analysis_splitter.setStretchFactor(1, 2)

        right_splitter.addWidget(self.analysis_splitter)
        
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
        # Match 'Dephasing Analysis' or old 'Relaxation Analysis' for backward compat logic if needed
        is_relax = mode_text.startswith("Relaxation") or mode_text.startswith("Dephasing") or is_global

        # 1. Manage Settings Visibility
        if hasattr(self, 'relax_settings_widget'):
            self.relax_settings_widget.setVisible(is_relax)
            
        if hasattr(self, 'adv_analysis_group'):
            # Show advanced analysis tools for BOTH Relaxation Analysis (Single) and Global Map
            self.adv_analysis_group.setVisible(is_relax)
            
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

    def request_analysis_update(self):
        """Called when settings change and we want to refresh the SINGLE peak analysis."""
        # Only if we are in Relaxation/Dephasing mode
        mode_text = self.combo_analysis_mode.currentText()
        if not (mode_text.startswith("Relaxation") or mode_text.startswith("Dephasing")):
            return
            
        # We need the last selected frequency. 
        # But we don't store it explicitly in a variable yet, except locally in on_pick.
        # Let's check if we can infer it or if we should store it.
        # Storing last_selected_freq is good practice.
        if hasattr(self, 'last_selected_freq') and self.last_selected_freq is not None:
             self.run_relaxation_analysis(self.last_selected_freq)

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

    def run_auto_phase(self):
        """
        Execute Auto-Phasing:
        1. Get unphased spectrum (run pipeline with p0=p1=0).
        2. Calculate entropy-minimized phase parameters.
        3. Update UI.
        """
        if self.raw_avg_data is None:
            QMessageBox.warning(self, "No Data", "Please load data first.")
            return

        self.statusBar().showMessage("Running Auto Phase (Minimum Entropy)...")
        QApplication.setOverrideCursor(Qt.WaitCursor)

        try:
            # 1. Get current params but reset phase to 0 to get raw spectrum
            # We want to find the phase of the *current* signal state (after SVD/Savgol etc)
            params = self._get_process_params()
            params['p0'] = 0
            params['p1'] = 0
            
            # 2. Get unphased spectrum
            # We can run process_fid in the main thread for this single operation 
            # as it shouldn't block for too long (SVD is already done or fast enough?)
            # If SVD is heavy, this might freeze UI.
            # But auto_phase itself takes time too. 
            # Ideally should be a Worker, but for simplicity let's try direct first.
            
            processor = Processor()
            _, raw_spec = processor.process_fid(self.raw_avg_data, params, self.loader_sampling_rate)
            
            # 3. Calculate optimal phase
            best_p0, best_p1 = processor.auto_phase_spectrum(raw_spec)
            
            # 4. Update UI
            # Block signals to prevent intermediate re-processing
            self.p0_slider.blockSignals(True)
            self.p1_slider.blockSignals(True)
            
            self.p0_slider.spinbox.setValue(float(best_p0))
            self.p1_slider.spinbox.setValue(float(best_p1))
            
            self.p0_slider.blockSignals(False)
            self.p1_slider.blockSignals(False)
            
            # Trigger final update (apply the new phase)
            self.run_processing()
            
            msg = f"Auto Phase Converged: p0={best_p0:.1f}°, p1={best_p1:.1f}°"
            self.statusBar().showMessage(msg)
            
        except Exception as e:
            QMessageBox.critical(self, "Auto Phase Failed", str(e))
            self.statusBar().showMessage("Auto Phase Error.")
        finally:
            QApplication.restoreOverrideCursor()

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
            
        # Time Axis with Start Offset (Truncation)
        trunc_start_pts = self.spin_trunc_start.value()
        t_offset = trunc_start_pts / self.loader_sampling_rate
        
        t = (np.arange(len(self.current_processed_time)) / self.loader_sampling_rate) + t_offset
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
        
        # 1. Update Plot 
        self.plot_spectrum_traffic_light() 
        
        # 2. Trigger Re-Analysis if "Analysis" tab is active and we have a selection
        # This implements "Global Data Mode": Change mode -> Change Analysis result
        self.request_analysis_update()

    def plot_spectrum_traffic_light(self, preserve_view=False):
        # Capture current Y-lim before clearing if preserving view
        current_ylim = self.ax_spec.get_ylim() if preserve_view else None
        
        self.ax_spec.clear()
        
        if self.current_freqs is None or self.current_spec is None:
            self.canvas_spec.draw()
            return

        # Determine data to plot based on view mode (Global Mode)
        spec_data = self.current_spec
        mode_label = "Magnitude"
        
        # Default keys for safety
        is_real = self.btn_view_real.isChecked() if hasattr(self, 'btn_view_real') else False
        is_imag = self.btn_view_imag.isChecked() if hasattr(self, 'btn_view_imag') else False
        is_abs = self.chk_view_abs.isChecked() if hasattr(self, 'chk_view_abs') else False

        if is_real:
            plot_data = np.real(spec_data)
            mode_label = "Real"
        elif is_imag:
            plot_data = np.imag(spec_data)
            mode_label = "Imaginary"
        else:
            plot_data = np.abs(spec_data)
            mode_label = "Magnitude"

        if is_abs:
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

    def update_spectrogram(self):
        """Compute and display Short-Time Fourier Transform"""
        import scipy.signal
        import scipy.fft
        from matplotlib.gridspec import GridSpec
        import matplotlib.pyplot as plt
        
        # 1. Check Data
        if self.current_processed_time is None:
            QMessageBox.warning(self, "No Data", "Please process data first.")
            return
            
        # 2. Switch Tab (Removed - now all visible)
        # self.tabs_analysis.setCurrentIndex(1) 
        
        # 3. Always work with COMPLEX data
        data_in = self.current_processed_time
        
        # 4. Get Parameters
        fs = self.loader_sampling_rate
        nperseg = self.combo_spec_window.currentData() 
        if not nperseg: nperseg = 1024
        nperseg = int(nperseg)
        noverlap = int(nperseg * 0.9) # 90% overlap
        
        try:
            # return_onesided=False ensures we get both Negative and Positive frequencies
            f, t, Zxx = scipy.signal.stft(
                data_in, 
                fs=fs, 
                window='hann', 
                nperseg=nperseg, 
                noverlap=noverlap, 
                return_onesided=False 
            )
            
            # Apply Time Offset (Truncation)
            trunc_start_pts = self.spin_trunc_start.value()
            t_offset = trunc_start_pts / fs
            t = t + t_offset
            
            f = scipy.fft.fftshift(f)
            Zxx = scipy.fft.fftshift(Zxx, axes=0)
            Sxx = np.abs(Zxx) 

            # --- Abs(Freq) Folding Logic ---
            # Default to Full Spectrum unless checked
            is_folded = False
            if hasattr(self, 'chk_spec_abs') and self.chk_spec_abs.isChecked():
                is_folded = True
                
            if is_folded:
                # Fold negative frequencies onto positive ones
                # Find index closest to 0
                idx_0 = np.argmin(np.abs(f)) 
                
                # Positive side (0 to +Nyquist)
                f_folded = f[idx_0:]
                S_folded = Sxx[idx_0:, :].copy()
                
                # Negative side (flipped)
                S_neg_flipped = np.flipud(Sxx[:idx_0, :])
                
                # Element-wise Max
                # Fix for shape mismatch error:
                # Ensure we don't write past the end of S_folded or read past end of S_neg_flipped.
                # Valid indices for writing S_folded start at 1. Max index is len(S_folded)-1.
                # So we can write at most len(S_folded) - 1 elements.
                limit = min(len(S_folded) - 1, len(S_neg_flipped))
                
                if limit > 0:
                     S_folded[1:limit+1] = np.maximum(S_folded[1:limit+1], S_neg_flipped[:limit])
                
                f = f_folded
                Sxx = S_folded
                mode_label = "Folded (|Freq|)"
            else:
                mode_label = "Full Spectrum"
                
            # Store STFT data for interactivity (T2* analysis)
            self.stft_f = f
            self.stft_t = t
            self.stft_Sxx = Sxx

        except Exception as e:
            QMessageBox.critical(self, "STFT Error", str(e))
            return
        
        # 6. Plot Layout (GridSpec)
        self.fig_stft.clear()
        
        # Define Grid: [Side Spectrum (15%), Spectrogram (85%)] swapped positions
        # Increased wspace slightly for labels
        gs = self.fig_stft.add_gridspec(1, 2, width_ratios=[1, 6], wspace=0.05)
        
        # Axis 1: Side Spectrum (Left)
        ax_side = self.fig_stft.add_subplot(gs[0])
        
        # Axis 2: Spectrogram (Right) - Share Y with Side Spectrum
        self.ax_stft = self.fig_stft.add_subplot(gs[1], sharey=ax_side)
        
        # Calculate Log Scale
        if self.chk_spec_log.isChecked():
            Sxx_dB = 20 * np.log10(Sxx + 1e-12)
            cmap_data = Sxx_dB
            cbar_label = "Amplitude (dB)"
            vmin = np.max(cmap_data) - 80 
            vmax = np.max(cmap_data)
            
            # Side Spectrum: Log of Mean
            mean_mag = np.mean(Sxx, axis=1)
            side_profile = 20 * np.log10(mean_mag + 1e-12)
        else:
            cmap_data = Sxx
            cbar_label = "Amplitude"
            vmin = 0
            vmax = np.max(cmap_data)
            side_profile = np.mean(Sxx, axis=1)

        # Plot Heatmap (Right)
        # We assume f and t are correct for pcolormesh
        mesh = self.ax_stft.pcolormesh(t, f, cmap_data, shading='auto', cmap='inferno', vmin=vmin, vmax=vmax)
        
        # Plot Side Profile (Left)
        # Plot Frequency on Y, Amplitude on X
        ax_side.plot(side_profile, f, 'k-', linewidth=0.8)
        fill_base = np.min(side_profile)
        ax_side.fill_betweenx(f, fill_base, side_profile, color='gray', alpha=0.3)
        
        # Styling Side Plot (The Y-axis Labels belong here now)
        ax_side.grid(True, alpha=0.3)
        unit_str = "(dB)" if self.chk_spec_log.isChecked() else ""
        ax_side.set_xlabel(f"Avg Amp {unit_str}")
        
        # Invert logic consideration:
        # Standard: 0 -> Max. User has 0 on left, Max on right. 
        # But for dB, -100 is left, -10 is right. This is correct.
        
        ax_side.set_xlim(auto=True)
        # Make sure X axis is normal
        if ax_side.get_xlim()[0] > ax_side.get_xlim()[1]: # if inverted
             ax_side.invert_xaxis()

        # Labels
        mode_str = " (|Hz|)" if is_folded else " (Hz)"
        ax_side.set_ylabel(f"Frequency{mode_str}")
        
        # Styling Heatmap (Remove Y Labels as they are on ax_side now)
        plt.setp(self.ax_stft.get_yticklabels(), visible=False)
        self.ax_stft.set_xlabel("Time (s)")
        self.ax_stft.set_title(f"Spectrogram (Win={nperseg})")

        # Limits
        f_min = self.freq_min.value()
        f_max = self.freq_max.value()
        
        if is_folded:
             if f_min < 0: f_min = 0
             
        # Set Limits on the Shared Axis (ax_side controls Y)
        ax_side.set_ylim(f_min, f_max)
        
        # Colorbar - attach to Spectrogram axis (Right side)
        self._cbar_stft = self.fig_stft.colorbar(mesh, ax=self.ax_stft, label=cbar_label)
        
        self.canvas_stft.draw()

    def on_spectrogram_click(self, event):
        """Handle clicks on Spectrogram/Side Profile to analyze T2* at that frequency"""
        # Check if Navigation Toolbar is active (Zoom/Pan mode)
        # If user is zooming/panning, we ignore the click for analysis
        if self.toolbar_stft.mode:
            return

        # Ensure click is within our axes (Main or Side)
        if event.inaxes not in self.fig_stft.axes:
            return
            
        # We care about Y coordinate (Frequency) as both axes share Frequency on Y
        target_freq = event.ydata
        
        if target_freq is None: 
            return
        
        # Check if we have data
        if not hasattr(self, 'stft_f') or self.stft_f is None: 
            return

        # Perform Analysis
        self.analyze_stft_t2(target_freq)

    def analyze_stft_t2(self, target_freq):
        from scipy.stats import linregress
        
        # 1. Find nearest frequency index
        # stft_f matches the rows of stft_Sxx
        idx = (np.abs(self.stft_f - target_freq)).argmin()
        actual_freq = self.stft_f[idx]
        
        # 2. Extract Data (Slice across time)
        times = self.stft_t
        amps = self.stft_Sxx[idx, :]
        
        # 3. Fit T2* (Exponential Decay)
        # Strategy: Log-Linear Fit on the "tail" or significant data
        
        # A. Peak Detection & Thresholding
        max_amp = np.max(amps)
        if max_amp == 0: return
        
        # Strategy: Ignore the "rising" part (from 0 to peak) typically caused by STFT windowing/startup
        # Start fitting from the maximum point onwards.
        idx_max = np.argmax(amps)
        
        # Create a mask for the "decay" portion only:
        # 1. Index >= idx_max (Post-peak)
        # 2. Amplitude > 10% of max (Avoid noise tail)
        mask_fit = np.zeros_like(amps, dtype=bool)
        
        # Only consider points after the peak
        decay_slice = amps[idx_max:]
        mask_decay = (decay_slice > max_amp * 0.1)
        
        # Map back to full array
        mask_fit[idx_max:] = mask_decay
        
        # B. Check if we have enough points
        if np.sum(mask_fit) < 4:
            self.statusBar().showMessage(f"Not enough data to fit T2* at {actual_freq:.1f}Hz")
            return
            
        t_fit = times[mask_fit]
        a_fit = amps[mask_fit]
        
        try:
            # ln(y) = ln(A) - (1/T2)*t
            # slope = -1/T2
            slope, intercept, r_val, p_val, std_err = linregress(t_fit, np.log(a_fit))
            
            if slope < 0:
                t2 = -1.0 / slope
            else:
                t2 = 0 # Growing signal?
                
            r2 = r_val**2
            
            # Generate Fit Line for Plotting (across full time range)
            fit_curve = np.exp(intercept + slope * times)
            
        except Exception as e:
            print(f"Fit Error: {e}")
            t2 = 0
            r2 = 0
            fit_curve = np.zeros_like(times)

        # 4. Switch to T2* Analysis Tab (Removed - now all visible)
        # self.tabs_analysis.setCurrentIndex(0) 
        
        # 5. Plot on ax_evo
        self.ax_evo.clear()
        
        # Plot Raw Data
        self.ax_evo.plot(times, amps, 'b-', alpha=0.5, label='STFT Magnitude')
        self.ax_evo.scatter(t_fit, a_fit, c='orange', s=10, zorder=3, label='Points used for Fit')
        
        # Plot Fit
        if t2 > 0:
            label_fit = f'Fit T2*={t2*1000:.1f}ms (R2={r2:.2f})'
            self.ax_evo.plot(times, fit_curve, 'r--', linewidth=2, label=label_fit)
        
        self.ax_evo.set_title(f"T2* Analysis @ {actual_freq:.1f} Hz (Spectrogram Slice)")
        self.ax_evo.set_xlabel("Time (s)")
        self.ax_evo.set_ylabel("Amplitude")
        self.ax_evo.legend()
        self.ax_evo.grid(True, alpha=0.3)
        
        # Store state for advanced tools (Envelope, etc.) if they want to use them later
        # We mimic the structure used by plot_relaxation_results
        self.current_decay_data = {
            'times': times,
            'amps': amps,
            'ax': self.ax_evo,
            'canvas': self.canvas_evo
        }
        
        self.canvas_evo.draw()
        self.statusBar().showMessage(f"Analyzed {actual_freq:.1f} Hz: T2* = {t2*1000:.1f} ms")
        
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
            freq = closest_row['Freq_Hz']
            self.last_selected_freq = freq # Store for re-runs
            
            current_mode = self.combo_analysis_mode.currentText()
            if current_mode.startswith("Relaxation") or current_mode.startswith("Dephasing"):
                self.run_relaxation_analysis(freq)
            else:
                self.plot_evolution(closest_row['Index'])
        df['dist'] = abs(df['Freq_Hz'] - click_x)
        closest_row = df.loc[df['dist'].idxmin()]
        
        # If reasonably close (e.g. 1Hz)
        if closest_row['dist'] < 5.0:
            current_mode = self.combo_analysis_mode.currentText()
            if current_mode.startswith("Relaxation") or current_mode.startswith("Dephasing"):
                self.run_relaxation_analysis(closest_row['Freq_Hz'])
            else:
                self.plot_evolution(closest_row['Index'])

    def run_relaxation_analysis(self, target_freq):
        if self.raw_avg_data is None:
            return
            
        self.ax_evo.clear()
        self.ax_evo.text(0.5, 0.5, "Calculating Dephasing...", ha='center', va='center')
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
        measure_mode = self.combo_measure_mode.currentText()
        
        self.relax_worker = RelaxationWorker(
            self.raw_avg_data, 
            target_freq, 
            sampling_rate,
            params,
            t_start=t_start,
            t_end=t_end,
            points=points,
            zero_fill_front=zero_fill_front,
            measure_mode=measure_mode
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
        self.statusBar().showMessage("Dephasing Analysis Complete")
        self.ax_evo.clear()
        
        # Save state for advanced analysis
        self.current_decay_data = {
            'times': np.array(times),
            'amps': np.array(amps),
            'ax': self.ax_evo,
            'canvas': self.canvas_evo
        }
        if hasattr(self, 'lbl_adv_result'):
            self.lbl_adv_result.setText("Analysis Ready")
        
        # Plot Data
        self.ax_evo.scatter(times, amps, c='b', alpha=0.6, label='Truncation Amp')
        
        # Plot Fit
        if len(fit_x) > 0:
            self.ax_evo.plot(fit_x, fit_y, 'r-', linewidth=2, label=f'Fit: T2*={t2*1000:.1f}ms')
            
        self.ax_evo.set_xlabel("Truncation Start Time (s)")
        self.ax_evo.set_ylabel("Peak Amplitude")
        self.ax_evo.set_title(f"Dephasing Analysis (R2={r2:.3f})")
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
        
        # CLEAR PREVIOUS RESULTS TO PREVENT STALE DATA DISPLAY
        self.batch_results_summary = None
        self.batch_results_details = None
        
        self.ax_evo.clear()
        self.ax_evo.text(0.5, 0.5, "Running Batch Analysis...", ha='center', va='center')
        self.canvas_evo.draw()
        
        # Global Data Mode Logic
        base_mode = "Magnitude"
        if hasattr(self, 'btn_view_real') and self.btn_view_real.isChecked(): base_mode = "Real"
        elif hasattr(self, 'btn_view_imag') and self.btn_view_imag.isChecked(): base_mode = "Imag"
        
        auto_phase = self.chk_auto_phase_step.isChecked() if hasattr(self, 'chk_auto_phase_step') else False
        
        if base_mode == "Real":
            if auto_phase:
                worker_mode = "Real (Auto-Phased)"
            else:
                worker_mode = "Real (Fixed Phase)"
        elif base_mode == "Magnitude":
            worker_mode = "Magnitude"
        else:
            worker_mode = "Magnitude"

        # Worker
        self.batch_worker = BatchRelaxationWorker(
            self.raw_avg_data,
            valid_peaks, 
            sampling_rate,
            params,
            t_start, t_end, points,
            zero_fill_front,
            use_tracking,
            track_win_hz,
            noise_threshold=self.peak_thr.value(), # Pass noise threshold
            measure_mode=worker_mode
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
        
        # Check if worker is running to prevent "No results" flash or stale data
        if hasattr(self, 'batch_worker') and self.batch_worker is not None and self.batch_worker.isRunning():
             self.ax_evo.clear()
             self.ax_evo.text(0.5, 0.5, "Running Batch Analysis...\nPlease Wait.", ha='center', va='center')
             self.canvas_evo.draw()
             return
        
        if self.batch_results_summary is None or self.batch_results_summary.empty:
             self.ax_evo.text(0.5, 0.5, "No results or Global Analysis not run", ha='center')
             self.canvas_evo.draw()
             return

        df = self.batch_results_summary
        
        # Determine Display Mode
        mode = self.combo_t2_display_mode.currentText() if hasattr(self, 'combo_t2_display_mode') else "Show Raw T2*"
        has_filtered = 'T2_star_filt_ms' in df.columns # Ensure filtered data exists
        
        if mode == "Show Overlay" and has_filtered:
            # 1. Overlay Mode: Raw (Ghost) + Filtered (Color)
            self.ax_evo.scatter(df['Freq_Hz'], df['T2_star_ms'], c='gray', alpha=0.3, s=20, label='Raw (Noisy)')
            
            sc = self.ax_evo.scatter(df['Freq_Hz'], df['T2_star_filt_ms'], c=df['R2_filt'], cmap='viridis', picker=5, s=40, edgecolors='k', label='Filtered')
            title = f"Global T2* Map (Overlay) (n={len(df)})"
            self.ax_evo.legend(fontsize='small')
            
        elif mode == "Show Filtered T2*" and has_filtered:
            # 2. Filtered Only
            sc = self.ax_evo.scatter(df['Freq_Hz'], df['T2_star_filt_ms'], c=df['R2_filt'], cmap='viridis', picker=5, s=40, edgecolors='k')
            title = f"Global T2* Map (Filtered) (n={len(df)})"
            
        else:
            # 3. Raw Only (Default or fallback if no filtered data)
            sc = self.ax_evo.scatter(df['Freq_Hz'], df['T2_star_ms'], c=df['R2'], cmap='viridis', picker=5, s=40, edgecolors='k')
            title = f"Global T2* Map (Raw) (n={len(df)})"
            
        self.ax_evo.set_xlabel('Frequency (Hz)')
        self.ax_evo.set_ylabel('T2* (ms)')
        self.ax_evo.set_title(title)
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
        self.current_detail_freq = freq # Cache for reference
        
        # Retrieve detail data
        data = self.batch_results_details.get(freq)
        if not data: return
        
        # Clean up existing twin axes to prevent stacking
        for other_ax in self.ax_detail.figure.axes:
             if other_ax is not self.ax_detail and other_ax.bbox.bounds == self.ax_detail.bbox.bounds:
                  self.ax_detail.figure.delaxes(other_ax)
        
        self.ax_detail.clear()
        
        times_ms = np.array(data['times']) * 1000
        amps = np.array(data['amps']) # Ensure numpy array
        freqs = data.get('freqs', None) # Get frequencies
        
        # Display Mode
        mode = self.combo_t2_display_mode.currentText() if hasattr(self, 'combo_t2_display_mode') else "Show Raw T2*"
        
        # Prepare Filtered Data
        amps_filt = data.get('amps_filt', None)
        has_filt = amps_filt is not None and len(amps_filt) > 0
        fit_x_filt = data.get('fit_x_filt', [])
        fit_y_filt = data.get('fit_y_filt', [])
        
        # Get T2 values
        t2_raw = data.get('t2', 0) * 1000.0
        t2_filt_val = 0
        if has_filt:
             # Find in summary
             if self.batch_results_summary is not None:
                  row = self.batch_results_summary[self.batch_results_summary['Freq_Hz'] == freq]
                  if not row.empty:
                       t2_filt_val = row.iloc[0].get('T2_star_filt_ms', 0)

        # Plotting Logic
        if mode == "Show Filtered T2*" and has_filt:
             self.ax_detail.plot(times_ms, amps_filt, 'o', color='gold', label='Filtered Data', markeredgecolor='k')
             if len(fit_x_filt) > 0:
                  self.ax_detail.plot(fit_x_filt*1000, fit_y_filt, 'g-', lw=2, label=f'Fit T2*={t2_filt_val:.0f}ms')
                  
        elif mode == "Show Overlay" and has_filt:
             self.ax_detail.plot(times_ms, amps, 'bo', alpha=0.2, label='Raw')
             self.ax_detail.plot(times_ms, amps_filt, 'o', color='gold', label='Filtered', markeredgecolor='k', markersize=4)
             
             fit_x_raw = data.get('fit_x', [])
             fit_y_raw = data.get('fit_y', [])
             if len(fit_x_raw) > 0:
                  self.ax_detail.plot(fit_x_raw*1000, fit_y_raw, 'r--', lw=1, alpha=0.5, label=f'Raw {t2_raw:.0f}ms')
             if len(fit_x_filt) > 0:
                  self.ax_detail.plot(fit_x_filt*1000, fit_y_filt, 'g-', lw=2, label=f'Filt {t2_filt_val:.0f}ms')

        else:
             # Default: Raw
             self.ax_detail.plot(times_ms, amps, 'bo', label='Data', alpha=0.6)
             fit_x_raw = data.get('fit_x', [])
             fit_y_raw = data.get('fit_y', [])
             if len(fit_x_raw) > 0:
                 self.ax_detail.plot(fit_x_raw*1000, fit_y_raw, 'r-', lw=2, label=f'Raw Fit T2*={t2_raw:.0f}ms')
        
        # Save state for advanced analysis
        self.current_decay_data = {
            'times': times_ms/1000, # Back to seconds
            'amps': amps,
            'freqs': freqs,
            'ax': self.ax_detail,
            'canvas': self.canvas_detail
        }

        # Plot Freq Drift on twin axis if available and tracking was used
        if freqs is not None and len(freqs) == len(times_ms):
            # Check if there is actual variation
            if np.std(freqs) > 0.01: 
                ax2 = self.ax_detail.twinx()
                ax2.plot(times_ms, freqs, 'k-', alpha=0.15, linewidth=1, label='Freq Drift')
                ax2.set_ylabel('Peak Center (Hz)', color='gray')
                ax2.tick_params(axis='y', labelcolor='gray')
        
        # Add Points info to title to explain resolution differences
        pts_setting = self.spin_relax_points.value() if hasattr(self, 'spin_relax_points') else '?'
        
        # Check if tracking was likely used
        is_tracking_data = False
        if freqs is not None and len(freqs) == len(amps) and len(freqs) > 1:
             if np.std(freqs) > 0.001: is_tracking_data = True
             
        track_str = " (Tracking ON)" if is_tracking_data else " (Static)"
        
        self.ax_detail.set_title(f"Decay @ {freq:.1f} Hz (Points={pts_setting}{track_str})")
        self.ax_detail.set_xlabel('Delay (ms)')
        self.ax_detail.set_ylabel('Amplitude')
        self.ax_detail.legend(fontsize='small')
        self.ax_detail.grid(True, alpha=0.3)
        self.canvas_evo.draw()
        self.ax_detail.set_title(f"Decay @ {freq:.1f} Hz")
        self.ax_detail.legend(fontsize='small')
        self.ax_detail.grid(True, linestyle=":", alpha=0.7)
        
        self.canvas_detail.draw()
        
        # Reset result label
        if hasattr(self, 'lbl_adv_result'):
            self.lbl_adv_result.setText("Advanced Analysis: Ready")

    def run_envelope_fit(self):
        if not hasattr(self, 'current_decay_data') or self.current_decay_data is None:
            QMessageBox.warning(self, "No Data", "Please run analysis or select a point first.")
            return

        # Unpack shared data
        times = self.current_decay_data['times']
        amps = self.current_decay_data['amps']
        ax = self.current_decay_data['ax']
        canvas = self.current_decay_data['canvas']
        
        # Remove previous Envelope Fit lines
        lines_to_remove = []
        for line in ax.get_lines():
            label = line.get_label()
            if label and (label.startswith('Env T2*') or label == 'Env Peaks'):
                lines_to_remove.append(line)
        for line in lines_to_remove:
            line.remove()
        
        # Calculate
        result = CurveFitter.fit_envelope(times, amps)
        
        if result['status'] != 'success':
            self.lbl_adv_result.setText(result['status'])
            return
            
        t2_env = result['t2']
        r2 = result['r2']
        
        # Plot Fit
        # Identify units scaling
        t_plot_scaled = result['t_plot'] * 1000 if ax == self.ax_detail else result['t_plot']
        
        ax.plot(t_plot_scaled, result['y_plot'], 
                'orange', linewidth=2, linestyle='--', label=f'Env T2*={t2_env*1000:.1f}ms')
        
        # Plot Peaks
        t_peaks_plot = result['peaks_t'] * 1000 if ax == self.ax_detail else result['peaks_t']
        ax.plot(t_peaks_plot, result['peaks_a'], 'rx', markersize=5, label='Env Peaks')
        
        # Update legend (hide Env Peaks)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if 'Env Peaks' in by_label: del by_label['Env Peaks']
        ax.legend(by_label.values(), by_label.keys(), fontsize='small')
        
        canvas.draw()
        
        self.lbl_adv_result.setText(f"Envelope T2*: {t2_env*1000:.1f} ms (R2={r2:.3f})")

    def run_cosine_fit(self):
        if not hasattr(self, 'current_decay_data') or self.current_decay_data is None:
            QMessageBox.warning(self, "No Data", "Please run analysis or select a point first.")
            return

        # Unpack shared data
        t = self.current_decay_data['times']
        y = self.current_decay_data['amps']
        ax = self.current_decay_data['ax']
        canvas = self.current_decay_data['canvas']

        # Remove old Beat Fits
        lines_to_remove = []
        for line in ax.get_lines():
            lbl = line.get_label()
            if lbl and lbl.startswith('Beat J='):
                lines_to_remove.append(line)
        for line in lines_to_remove:
            line.remove()
        
        # Calculate
        result = CurveFitter.fit_damped_cosine(t, y)
        
        if result['status'] != 'success':
             self.lbl_adv_result.setText(result['status'])
             return
             
        J_fit = result['J']
        T2_fit = result['T2']
            
        # Plot
        t_plot_scaled = result['t_plot'] * 1000 if ax == self.ax_detail else result['t_plot']
        ax.plot(t_plot_scaled, result['y_plot'], 'g-', linewidth=2, alpha=0.8, label=f'Beat J={J_fit:.1f}Hz')
        
        # Update legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if 'Env Peaks' in by_label: del by_label['Env Peaks']
        ax.legend(by_label.values(), by_label.keys(), fontsize='small')
        
        canvas.draw()
        
        self.lbl_adv_result.setText(f"Beat Fit: J={J_fit:.2f} Hz | T2*={T2_fit*1000:.1f} ms")


    def run_oscillation_filter(self):
        if not hasattr(self, 'current_decay_data') or self.current_decay_data is None:
            QMessageBox.warning(self, "No Data", "Please run analysis or select a point first.")
            return
            
        # Unpack shared data
        t = self.current_decay_data['times']
        y = self.current_decay_data['amps']
        ax = self.current_decay_data['ax']
        canvas = self.current_decay_data['canvas']
        
        # Get Current Noise Threshold as C limit
        # Logic: C (Baseline) should not exceed the detection threshold (which is > Noise)
        noise_est = self.peak_thr.value()
        
        # Calculate
        result = CurveFitter.remove_oscillation_fft(t, y, noise_level=noise_est)
        
        if result['status'] != 'success':
            self.lbl_adv_result.setText(result['status'])
            return
            
        y_clean = result.get('amps_filtered')
        f_osc = result.get('f_osc', 0)
        # fs_decay seems unused or legacy. If needed, we can get it from f_osc or just ignore.
        
        fit_res = result.get('fit_result', {})
        
        # Remove previous 'Filtered' and its fit lines
        lines_to_remove = []
        for line in ax.get_lines():
            lbl = line.get_label()
            # ONLY remove lines that are specifically from the Filter tool
            # 'Filtered' (Cyan) or 'Filtered Fit ...' (Red dashed)
            if lbl and (lbl == 'Filtered' or lbl.startswith('Filtered Fit')):
                lines_to_remove.append(line)
        for line in lines_to_remove: line.remove()
        
        # Plot Filtered Signal
        t_plot = t * 1000 if ax == self.ax_detail else t
        ax.plot(t_plot, y_clean, 'c-', linewidth=2.5, label='Filtered')
        
        # Plot Filtered Fit
        if fit_res.get('status') == 'success':
            y_fit = fit_res['y_fit']
            # Improved Label to identify it's from the filter
            label_fit = f"Filtered Fit T2*={fit_res['T2']*1000:.1f}ms"
            ax.plot(t_plot, y_fit, 'r--', linewidth=2.0, label=label_fit)
            
        # Deduplicate Legend
        # Grab all unique labels in order
        handles, labels = ax.get_legend_handles_labels()
        unique = {}
        for h, l in zip(handles, labels):
            if l not in unique and l != 'Env Peaks': # Ignore Env Peaks in legend
                unique[l] = h
        
        ax.legend(unique.values(), unique.keys(), fontsize='small')
        
        canvas.draw()
        
        status_text = f"Filtered {f_osc:.1f} Hz."
        if fit_res.get('status') == 'success':
            status_text += f" Filtered T2*={fit_res['T2']*1000:.1f}ms (R2={fit_res['R2']:.3f})"
        
        self.lbl_adv_result.setText(status_text)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
