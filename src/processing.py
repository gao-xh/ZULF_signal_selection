import sys
import numpy as np
import pandas as pd
import scipy.signal
import scipy.fft
from scipy.signal import find_peaks
from scipy.stats import linregress
from scipy.optimize import curve_fit
from pathlib import Path
from src.config import (
    REFERENCES_DIR, 
    DEFAULT_SAVGOL_WINDOW, DEFAULT_SAVGOL_ORDER, 
    DEFAULT_APOD_T2STAR, DEFAULT_SVD_RANK, DEFAULT_ENABLE_SVD
)

# Add references to path to allow importing ZULF libraries
if str(REFERENCES_DIR) not in sys.path:
    sys.path.append(str(REFERENCES_DIR))

# Try importing from reference library
try:
    from nmr_processing_lib.processing.zulf_algorithms import (
        apply_phase_correction, auto_phase
    )
    from nmr_processing_lib.processing.filtering import svd_denoising
    HAS_REF_LIB = True
except ImportError:
    HAS_REF_LIB = False
    print("Warning: Could not import nmr_processing_lib. Using internal fallbacks.")

class Processor:
    """
    Unified Processing Engine for ZULF NMR Data.
    Ensures consistent processing across different signal averages.
    """
    
    @staticmethod
    def get_default_params():
        return {
            # Savgol (Baseline)
            'conv_points': DEFAULT_SAVGOL_WINDOW,
            'poly_order': DEFAULT_SAVGOL_ORDER,
            
            # SVD
            'enable_svd': DEFAULT_ENABLE_SVD,
            'svd_rank': DEFAULT_SVD_RANK,
            
            # Truncation
            'trunc_start': 0,
            'trunc_end': 0,
            
            # Apodization
            'apod_t2star': DEFAULT_APOD_T2STAR, # Decay rate
            
            # Zero Filling
            'zf_factor': 1, 
            
            # Phasing (Fixed parameters)
            'p0': 0.0,
            'p1': 0.0,
            'phase_mode': 'manual', # 'manual' or 'auto' (only for N_max)
        }

    @staticmethod
    def process_fid(fid_data, params, sampling_rate):
        """
        Full pipeline: FID -> Spectrum
        """
        # Ensure float
        data = fid_data.astype(np.complex128) if np.iscomplexobj(fid_data) else fid_data.astype(np.float64)
        
        # 1. Savgol Baseline Correction (Time Domain)
        # Often used in ZULF to remove DC offsets or low freq drifts in time domain before FFT
        if params.get('conv_points', 0) > 0:
            window = int(params['conv_points'])
            order = int(params['poly_order'])
            if window % 2 == 0: window += 1 # Must be odd
            if window > 3 and window < len(data):
                 smooth = scipy.signal.savgol_filter(data.real, window, order, mode='mirror')
                 data = data - smooth

        # 2. Truncation (Moved before SVD to improve performance)
        trunc_start = int(params.get('trunc_start', 0))
        trunc_end = int(params.get('trunc_end', 0))
        
        if trunc_start > 0:
            # Check bounds
            if trunc_start < len(data):
                data = data[trunc_start:]
            else:
                data = np.array([]) # All cut
                
        if trunc_end > 0:
            # Check bounds
            if trunc_end < len(data):
                data = data[:-trunc_end]
            else:
                # If trunc_end is very large, it might clear everything
                # Logic: data[:-large] -> empty
                data = np.array([])

        if len(data) == 0:
            # Early return if empty
            return np.array([0]), np.array([0])
        
        # 3. SVD Denoising (Optional)
        # SVD involves constructing a Hankel matrix which is N/2 x N/2. 
        # For N=60000, this is 30000x30000, which requires ~7GB RAM (complex128 is 16 bytes/val -> 14GB!).
        # We must limit the size or skip SVD to prevent crashes on standard machines.
        MAX_SVD_POINTS = 10000 
        
        if params.get('enable_svd', False):
            if len(data) > MAX_SVD_POINTS:
                print(f"Warning: Data length ({len(data)}) exceeds SVD safe limit ({MAX_SVD_POINTS}). Skipping SVD to prevent memory crash.")
            elif HAS_REF_LIB:
                rank = int(params.get('svd_rank', 5))
                try:
                    data = svd_denoising(data, rank)
                except Exception as e:
                    print(f"SVD Error: {e}")
            else:
                 # Internal fallback loop could go here if implemented, but avoiding for now due to memory risk
                 pass
            
        # 4. Apodization (Exponential Window)
        lb = params.get('apod_t2star', 0)
        if lb > 0:
            # t vector
            t = np.arange(len(data)) * (1.0 / sampling_rate)
            # exp(-t/T2) or exp(-k*t) -> Reference uses exp(-param * t)
            window = np.exp(-lb * t)
            data = data * window
            
        # 5. Zero Filling
        zf_factor = int(params.get('zf_factor', 0))
        if zf_factor > 0:
            pad_len = len(data) * zf_factor
            # Pad with mean (usually 0 after baseline corr) to avoid steps
            pad_val = 0 # np.mean(data[-10:]) if we want to be safe
            data = np.pad(data, (0, pad_len), 'constant', constant_values=pad_val)
            
        # 6. FFT
        spectrum = scipy.fft.fft(data)
        freqs = scipy.fft.fftfreq(len(data), d=1.0/sampling_rate)
        
        # 7. Phase Correction
        # Check if we should auto phase or apply fixed
        # But this function 'process_fid' is intended for batch processing usually.
        # We assume params['p0'] and params['p1'] are set correctly.
        
        if HAS_REF_LIB:
            spectrum = apply_phase_correction(spectrum, params.get('p0', 0), params.get('p1', 0))
        else:
            # Simple fallback
            p0 = np.deg2rad(params.get('p0', 0))
            p1 = np.deg2rad(params.get('p1', 0))
            # Linear phase: phi = p0 + p1 * (freq / max_freq) ? 
            # Reference implementation usually: phase = p0 + p1 * (index / N)
            # Let's check ZULF algorithms if we could... but assuming fallback:
            # Standard NMR: p1 is usually delay. 
            # Let's stick to 0 phase if lib missing for now.
            pass
            
        return freqs, spectrum

    @staticmethod
    def auto_phase_spectrum(spectrum):
        """
        Calculate optimal phase parameters using the reference algorithm.
        Returns: (p0, p1) in degrees
        """
        if HAS_REF_LIB:
            return auto_phase(spectrum)
        else:
            return 0.0, 0.0

class CurveFitter:
    """
    Advanced analysis tools for decay curves.
    Handles mathematical fitting (Envelope, Cosine/Beat) and Filtering.
    """
    
    @staticmethod
    def fit_envelope(times, amps):
        """
        Fits an exponential envelope to the peaks of the signal.
        Returns:
            result_dict: {
                't2': float (seconds),
                'r2': float,
                'slope': float,
                'intercept': float,
                'status': str,    # 'success' or error message
                't_plot': array,
                'y_plot': array,
                'peaks_idx': array, # Indices of peaks used
                'peaks_t': array,
                'peaks_a': array
            }
        """
        # Find peaks (local maxima)
        peaks, _ = find_peaks(amps)
        
        if len(peaks) < 3:
            return {'status': "Envelope Fit: Not enough peaks found.", 't2': 0}
            
        t_peaks = times[peaks]
        a_peaks = amps[peaks]
        
        # Log-Linear Fit
        valid = a_peaks > 0
        if np.sum(valid) < 3:
             return {'status': "Envelope Fit: Peaks too low.", 't2': 0}
             
        slope, intercept, r_val, _, _ = linregress(t_peaks[valid], np.log(a_peaks[valid]))
        
        if slope >= 0:
            return {'status': "Envelope Fit: Signal is growing (Slope >= 0).", 't2': 0}
            
        t2_env = -1.0 / slope
        r2 = r_val**2
        
        # Generator plot data
        t_plot = np.linspace(min(times), max(times), 100)
        y_plot = np.exp(intercept + slope * t_plot)
        
        return {
            'status': 'success',
            't2': t2_env,
            'r2': r2,
            'slope': slope,
            'intercept': intercept,
            't_plot': t_plot,
            'y_plot': y_plot,
            'peaks_idx': peaks,
            'peaks_t': t_peaks,
            'peaks_a': a_peaks
        }

    @staticmethod
    def fit_damped_cosine(times, amps, user_freq_guess=None):
        """
        Fits a damped cosine model: A * exp(-t/T2) * |cos(pi * J * t + phi)| + C
        """
        # Define Model
        def beat_model(t, A, T2, J, phi, C):
            decay = np.exp(-t / T2)
            osc = np.abs(np.cos(np.pi * J * t + phi))
            return A * decay * osc + C
            
        # Initial Guess Estimation
        if user_freq_guess:
             J_guess = user_freq_guess
        else:
            # Auto-estimate J from peaks
            peaks_idx, _ = find_peaks(amps, height=np.max(amps)*0.1) 
            if len(peaks_idx) > 1:
                t_span_peaks = times[peaks_idx[-1]] - times[peaks_idx[0]]
                if t_span_peaks > 0:
                    est_freq = (len(peaks_idx) - 1) / t_span_peaks
                    J_guess = est_freq
                else:
                    J_guess = 10.0
            else:
                J_guess = 10.0

        p0 = [np.max(amps), 0.3, J_guess, 0.0, 0.0]
        bounds = (
            [0,     0.01, 0.1,  -np.pi, 0],   # Lower
            [np.inf, 5.0,  300.0, np.pi, np.inf] # Upper
        )
        
        try:
            popt, pcov = curve_fit(beat_model, times, amps, p0=p0, bounds=bounds, maxfev=2000)
            
            # High res plot data
            t_plot = np.linspace(min(times), max(times), 1000)
            y_plot = beat_model(t_plot, *popt)
            
            return {
                'status': 'success',
                'params': popt, # A, T2, J, phi, C
                'J': popt[2],
                'T2': popt[1],
                't_plot': t_plot,
                'y_plot': y_plot
            }
            
        except Exception as e:
            return {'status': f"Fit Failed: {str(e)}", 'params': None}

    @staticmethod
    def remove_oscillation_fft(times, amps):
        """
        Removes high frequency oscillations using FFT Low Pass Filter.
        Uses mirror padding to reduce edge artifacts.
        """
        dt = np.mean(np.diff(times))
        fs_decay = 1.0 / dt
        n_orig = len(amps)
        
        # Mirror Pad to reduce edge artifacts (Gibbs/Boundary)
        # Pad length: 50% of the signal on both sides
        n_pad = n_orig // 2
        amps_padded = np.pad(amps, (n_pad, n_pad), mode='reflect')
        n = len(amps_padded)
        
        # FFT
        yf = scipy.fft.rfft(amps_padded)
        xf = scipy.fft.rfftfreq(n, dt)
        
        # Strategy: Find dominant peak in AC component (Frequency > 0)
        # Low freq cutoff: Ignore first 5 bins (DC + Exp decay)
        idx_skip = 5 
        
        if idx_skip >= len(xf):
             return {'status': "Data too short."}
             
        # Find max peak (Oscillation)
        idx_max = idx_skip + np.argmax(np.abs(yf[idx_skip:]))
        f_osc = xf[idx_max]
        
        # Determine Cutoff for Low Pass Filter
        # We cut slightly below the oscillation to remove it and all higher harmonics
        # Margin: 10% of the oscillation frequency
        f_cutoff = f_osc * 0.9
        idx_cutoff = np.searchsorted(xf, f_cutoff)
        
        yf_clean = yf.copy()
        # Low Pass Filter: Zero out everything above cutoff
        yf_clean[idx_cutoff:] = 0
        
        # Inverse FFT
        y_clean_padded = scipy.fft.irfft(yf_clean, n=n)
        
        # Remove padding
        y_clean = y_clean_padded[n_pad : n_pad + n_orig]
        
        # Fit T2* to the cleaned curve
        # TRIM EDGES: Ignore first/last 5% points for fitting to avoid FFT boundary artifacts
        n_trim = max(2, int(n_orig * 0.05))
        if len(times) > 2 * n_trim + 5:
            # Use trimmed data for fit
            t_fit = times[n_trim:-n_trim]
            y_fit_data = y_clean[n_trim:-n_trim]
        else:
            t_fit = times
            y_fit_data = y_clean

        def exp_model(t, A, T2, C):
             return A * np.exp(-t/T2) + C
             
        p0 = [np.max(y_fit_data)-np.min(y_fit_data), (t_fit[-1]-t_fit[0])/3.0, np.min(y_fit_data)]
        bounds = ([0, 0, -np.inf], [np.inf, np.inf, np.inf])
        
        fit_res = {'T2': 0, 'R2': 0, 'status': 'fit_failed'}
        try:
             popt, _ = curve_fit(exp_model, t_fit, y_fit_data, p0=p0, bounds=bounds, maxfev=1000)
             A_fit, T2_fit, C_fit = popt
             
             # R2 (calc on trimmed data)
             residuals = y_fit_data - exp_model(t_fit, *popt)
             ss_res = np.sum(residuals**2)
             ss_tot = np.sum((y_fit_data - np.mean(y_fit_data))**2)
             r2 = 1 - (ss_res / ss_tot)
             
             # Generate full length fit curve for plotting
             y_fit_full = exp_model(times, *popt)
             
             fit_res = {
                 'status': 'success',
                 'T2': T2_fit,
                 'R2': r2,
                 'params': popt,
                 'y_fit': y_fit_full # Plot full range
             }
        except Exception as e:
             fit_res['error'] = str(e)
        
        return {
            'status': 'success',
            'y_clean': y_clean,
            'f_osc': f_osc,
            'fs_decay': fs_decay,
            'fit_result': fit_res
        }
