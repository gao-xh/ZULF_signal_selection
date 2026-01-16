import sys
import numpy as np
import pandas as pd
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QGroupBox, QFileDialog, QSplitter, QProgressBar, QMessageBox,
    QTabWidget
)
from PySide6.QtCore import Qt, QThread, Signal, Slot

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from src.ui_components import SliderSpinBox
from src.loader import ProgressiveLoader
from src.processing import Processor
from src.validator import SignalValidator

class ValidationWorker(QThread):
    finished = Signal(object, object, object, object) # results_df, freqs, spec, evo_data
    error = Signal(str)
    progress = Signal(str)

    def __init__(self, folder, params):
        super().__init__()
        self.folder = folder
        self.params = params

    def run(self):
        try:
            self.progress.emit("Loading Data...")
            loader = ProgressiveLoader(self.folder)
            processor = Processor()
            validator = SignalValidator(loader, processor)
            
            self.progress.emit("Running Progressive Validation (This may take a while)...")
            results, freqs, spec, evo_data = validator.run_validation(self.params)
            
            self.finished.emit(results, freqs, spec, evo_data)
            
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ZULF Signal Selection - Progressive Validator")
        self.resize(1200, 800)
        
        self.current_results = None
        self.current_freqs = None
        self.current_spec = None
        
        self._setup_ui()
        
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        # --- Left Panel: Controls ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setFixedWidth(300)
        
        # 1. Processing Controls
        proc_group = QGroupBox("Processing Parameters")
        proc_layout = QVBoxLayout()
        
        self.savgol_window = SliderSpinBox("Savgol Window", 1, 101, 31, step=2)
        proc_layout.addWidget(self.savgol_window)
        
        self.savgol_order = SliderSpinBox("Savgol Order", 1, 6, 3)
        proc_layout.addWidget(self.savgol_order)
        
        self.apod_rate = SliderSpinBox("Apod T2* (s)", 0.01, 1.0, 0.05, is_float=True, decimals=3, step=0.01)
        proc_layout.addWidget(self.apod_rate)

        # Fixed Phase
        self.p0_slider = SliderSpinBox("Phase 0", -180, 180, 0, is_float=True)
        proc_layout.addWidget(self.p0_slider)
        
        proc_group.setLayout(proc_layout)
        left_layout.addWidget(proc_group)
        
        # 2. Validation Thresholds
        val_group = QGroupBox("Decision Thresholds")
        val_layout = QVBoxLayout()
        
        self.thr_r2 = SliderSpinBox("Min R2 Score", 0.0, 1.0, 0.8, is_float=True)
        val_layout.addWidget(self.thr_r2)
        self.thr_r2.valueChanged.connect(self.update_verdicts)
        
        self.thr_slope = SliderSpinBox("Min Slope", -0.5, 2.0, 0.1, is_float=True)
        val_layout.addWidget(self.thr_slope)
        self.thr_slope.valueChanged.connect(self.update_verdicts)
        
        val_group.setLayout(val_layout)
        left_layout.addWidget(val_group)
        
        # Action Buttons
        self.btn_load = QPushButton("Load Folder & Run Analysis")
        self.btn_load.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.btn_load.clicked.connect(self.start_analysis)
        left_layout.addWidget(self.btn_load)
        
        self.progress_bar = QProgressBar()
        left_layout.addWidget(self.progress_bar)
        
        left_layout.addStretch()
        main_layout.addWidget(left_panel)
        
        # --- Right Panel: Visualization ---
        right_splitter = QSplitter(Qt.Vertical)
        
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
        
    def start_analysis(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Experiment Folder")
        if not folder:
            return
            
        params = {
            'conv_points': int(self.savgol_window.value()) if int(self.savgol_window.value()) % 2 != 0 else int(self.savgol_window.value()) + 1,
            'poly_order': int(self.savgol_order.value()),
            'apod_t2star': float(self.apod_rate.value()),
            'p0': float(self.p0_slider.value()),
            'phase_mode': 'auto', # Auto for N_max by default
            'trunc_start': 0,
            'enable_svd': True # Default enable for better quality
        }
        
        self.btn_load.setEnabled(False)
        self.progress_bar.setRange(0, 0) # Indeterminate
        
        self.worker = ValidationWorker(folder, params)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.error.connect(self.on_error)
        self.worker.progress.connect(self.update_status)
        self.worker.start()
        
    def update_status(self, msg):
        self.statusBar().showMessage(msg)
        
    def on_error(self, msg):
        self.btn_load.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        self.statusBar().showMessage("Error")
        QMessageBox.critical(self, "Analysis Failed", msg)
        
    def on_analysis_finished(self, results, freqs, spec, evo_data):
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
        self.plot_spectrum_traffic_light()
        
    def plot_spectrum_traffic_light(self):
        self.ax_spec.clear()
        
        # Plot Magnitude Spectrum
        freqs = self.current_freqs
        mag = np.abs(self.current_spec)
        self.ax_spec.plot(freqs, mag, 'k-', linewidth=0.8, alpha=0.6, label='Spectrum (N_max)')
        
        # Plot Candidates
        df = self.current_results
        
        signals = df[df['Verdict'] == True]
        noise = df[df['Verdict'] == False]
        
        # Plot Green Signals
        if not signals.empty:
            self.ax_spec.scatter(
                signals['Freq_Hz'], 
                mag[signals['Index'].astype(int)], 
                c='g', s=100, marker='o', alpha=0.8, edgecolors='k', 
                label='Signal (Validated)', picker=5
            )
            
        # Plot Red Noise
        if not noise.empty:
            self.ax_spec.scatter(
                noise['Freq_Hz'], 
                mag[noise['Index'].astype(int)], 
                c='r', s=50, marker='x', alpha=0.6, 
                label='Noise', picker=5
            )
            
        self.ax_spec.set_title("Spectral Validation (Click points for details)")
        self.ax_spec.set_xlabel("Frequency (Hz)")
        self.ax_spec.set_ylabel("Amplitude")
        self.ax_spec.legend()
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
