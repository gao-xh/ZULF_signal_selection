import sys
import numpy as np
import pandas as pd
import scipy.signal
import scipy.fft
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import find_peaks
from scipy.stats import linregress
from scipy.optimize import curve_fit
from pathlib import Path
from src.config import (
    REFERENCES_DIR, 
    DEFAULT_SAVGOL_WINDOW, DEFAULT_SAVGOL_ORDER, 
    DEFAULT_APOD_T2STAR, DEFAULT_SVD_RANK, DEFAULT_ENABLE_SVD
)

# Import new auto-phase module
from src.auto_phase import apply_phase_correction, auto_phase_entropy

# Add references to path to allow importing ZULF libraries
if str(REFERENCES_DIR) not in sys.path:
    sys.path.append(str(REFERENCES_DIR))

# Try importing from reference library
try:
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

            # Baseline Correction (ASLS)
            'baseline_enable': False,
            'baseline_lambda': 1000.0,
            'baseline_p': 0.01,
            'baseline_niter': 10,
        }
        
    @staticmethod
    def asls_baseline(y, lam, p, niter=10):
        """
        Asymmetric Least Squares baseline correction.
        y : 1D array (signal)
        lam : float, smoothing parameter (larger => smoother baseline)
        p : float between 0 and 1, asymmetry parameter (smaller => baseline stays below peaks)
        niter : iterations
        Returns baseline (same length as y)
        """
        L = len(y)
        # Create difference matrix
        D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L-2, L))
        # D.T @ D
        DTD = D.T @ D
        
        w = np.ones(L)
        z = np.zeros(L)
        
        for i in range(int(niter)):
            W = sparse.diags(w, 0)
            Z = W + lam * DTD
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y <= z)
            
        return z

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

        # 2. Time Domain Phase Correction (P0 & P1/Shift) - Blake's Method
        # In this mode, P0 is a constant phase multiply, and P1 is a time-shift.
        # This replaces the frequency domain correction.
        
        # P0: Constant Phase (Degrees -> Radians)
        p0_deg = params.get('p0', 0)
        if abs(p0_deg) > 1e-6:
            data = data * np.exp(1j * np.deg2rad(p0_deg))
        
        # P1: Time Shift (Points)
        # Note: UI 'p1' (Phase 1 deg) is repurposed here as 'Time Shift (points)'
        # Positive = Delay (prepend), Negative = Advance (cut end)
        shift_pts = int(params.get('p1', 0))
        if shift_pts != 0:
            avg_val = np.mean(data) # Blake uses mean for padding
            if shift_pts > 0:
                # Prepend 'shift_pts' copies of mean, push signal to right
                prepend = np.ones(shift_pts, dtype=data.dtype) * avg_val
                # To maintain length: Prepend and cut end? 
                # Blake's code: "prepend... np.concatenate" then "Truncation (from beginning)" later?
                # Actually Blake's code: 
                # if shift > 0: prepend_mean(..., shift, ...) -> Data grows.
                # But typically we want to keep array size consistent or let it grow if allowed.
                # Standard practice: Keep length or grow. Here we grow, FFT handles it.
                data = np.concatenate((prepend, data))
            elif shift_pts < 0:
                # Negative shift: Cut the end?
                # Blake: "savgol_td = savgol_td[-shift:]" -> This slices FROM THE END?
                # No, [-neg:] means grabbing from index.
                # Example: shift = -14. savgol_td[-(-14):] -> savgol_td[14:]
                # This cuts the START. 
                # Wait, checking Blake's: "savgol_td = savgol_td[-int(float(start_point_shift)):]"
                # If shift = -14, this is savgol_td[14:]. This removes the FIRST 14 points (Shift Left / Advance).
                slice_idx = -shift_pts
                if slice_idx < len(data):
                    data = data[slice_idx:]
                else:
                    data = np.array([])

        # 3. Truncation & Trace Operations
        # Front Truncation (Start)
        trunc_start = int(params.get('trunc_start', 0))
        
        # 3. Truncation & Trace Operations
        # Front Truncation (Start)
        trunc_start = int(params.get('trunc_start', 0))
        
        if trunc_start > 0:
            # Determine Fill Mode
            fill_mode = params.get('trunc_fill_mode', 'cut')
            # Backward compatibility for 'zero_fill_front' boolean
            if params.get('zero_fill_front', False) and fill_mode == 'cut':
                fill_mode = 'zero'
            
            if fill_mode == 'zero':
                # Zero-Fill Front Mode: MUTE the first N points
                if trunc_start < len(data):
                    fill_val = 0 
                    data[:trunc_start] = fill_val
            
            elif fill_mode == 'harmonic':
                # Harmonic Fill Mode: Fill with Sine Wave
                if trunc_start < len(data):
                    fill_freq = float(params.get('trunc_fill_freq', 60.0))
                    
                    # Generate time vector for the fill region
                    # Time runs from 0 to trunc_start/fs
                    t_fill = np.arange(trunc_start) / sampling_rate
                    
                    # --- Continuity Optimization ---
                    # We want the wave to seamlessly connect to data[trunc_start]
                    # So f(t_end) should match data[trunc_start] in phase and amplitude.
                    
                    # 1. Get the target value at the junction
                    target_val = data[trunc_start]
                    
                    # 2. Extract Amplitude (A) & Phase (phi)
                    if np.iscomplexobj(data):
                        A = np.abs(target_val)
                        phi_junction = np.angle(target_val)
                    else:
                        # For real data, we can't fully determine Phase & Amplitude from one point 
                        # without making assumptions. 
                        # Assumption: The user wants a sinewave that *Passes Through* target_val
                        # BUT has an amplitude consistent with local noise/signal.
                        # If we just pick A=|target_val|, we risk A being near zero (zero crossing).
                        # Better strategy: 
                        #   Estimate Amplitude A from local region (e.g. 50pts RMS * sqrt(2))
                        #   Calculate Phase phi such that A*cos(phi) = target_val
                        ref_slice = data[trunc_start:trunc_start+50] if (trunc_start+50 < len(data)) else data[trunc_start:]
                        if len(ref_slice) > 0:
                            local_rms = np.std(ref_slice)
                            A = local_rms * 1.414 # Estimate Peak Amp from RMS
                            # Ensure A is at least large enough to cover target_val
                            if A < abs(target_val): A = abs(target_val) * 1.1 
                        else:
                            A = abs(target_val)
                        
                        if A == 0: A = 1.0
                        
                        # Calculate required phase at junction: target = A * cos(theta)
                        # theta = arccos(target/A)
                        # We have ambiguity (plus/minus), pick positive slope?
                        # Let's just pick one.
                        # Clamp ratio to [-1, 1]
                        ratio = target_val / A
                        ratio = max(-1.0, min(1.0, ratio))
                        phi_junction = np.arccos(ratio)
                        
                    # 3. Calculate Phase Offset (Start Phase)
                    # We want: theta(t_end) = 2*pi*f*t_end + phi_0 = phi_junction
                    # => phi_0 = phi_junction - 2*pi*f*t_end
                    t_end = trunc_start / sampling_rate
                    phi_0 = phi_junction - (2 * np.pi * fill_freq * t_end)
                    
                    # 4. Generate the wave
                    # t_fill runs 0 to t_end
                    arg = (2 * np.pi * fill_freq * t_fill) + phi_0
                    
                    if np.iscomplexobj(data):
                        wave = A * np.exp(1j * arg)
                    else:
                        wave = A * np.cos(arg)
                    
                    # 5. Apply Taper: Linear fade-in from 0 to 1
                    # This avoids a click at t=0 while preserving full continuity at t=end
                    if trunc_start > 0:
                        taper = np.linspace(0.0, 1.0, trunc_start)
                        wave = wave * taper
                    
                    data[:trunc_start] = wave
                    
            else:
                # Standard Truncate Mode: CUT the first N points
                if trunc_start < len(data):
                    data = data[trunc_start:]
                else:
                    data = np.array([])

        # End Truncation
        trunc_end = int(params.get('trunc_end', 0))
        if trunc_end > 0:
            # Note: Blake cuts from end differently based on his specific flow, 
            # likely removing acquisition decay or bad end points.
            if trunc_end < len(data):
                data = data[:-trunc_end]
            else:
                 data = np.array([])

        if len(data) == 0:
            # Early return if empty
            return np.array([0]), np.array([0])
        
        # 4. SVD Denoising (Optional)
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
        
        # 7. Phase Correction (Frequency Domain - DISABLED)
        # Using Blake's Method (Time Domain) strictly.
        # However, if 'apply_phase_correction' is called, it might double apply.
        # Let's ensure P0/P1 are zeroed out if we passed them?
        # No, we already applied them in Time Domain above.
        # We assume parameters 'p0' and 'p1' were consumed there.
        # To be safe, we DO NOT call apply_phase_correction here again.
        
        # spectrum = apply_phase_correction(spectrum, params.get('p0', 0), params.get('p1', 0))
        
        # 8. Baseline Correction (ASLS)
        if params.get('baseline_enable', False):
            # Baseline correction is typically performed on the REAL part of the phased spectrum
            lam = float(params.get('baseline_lambda', 1000))
            p_val = float(params.get('baseline_p', 0.01))
            niter = int(params.get('baseline_niter', 10))
            
            real_part = np.real(spectrum)
            
            try:
                # Calculate baseline
                baseline = Processor.asls_baseline(real_part, lam, p_val, niter)
                # Subtract baseline from Real part
                corrected_real = real_part - baseline
                # Reconstruct complex spectrum (Imaginary part untouched)
                spectrum = corrected_real + 1j * np.imag(spectrum)
            except Exception as e:
                print(f"Baseline Correction Error: {e}")
            
        return freqs, spectrum

    @staticmethod
    def auto_phase_spectrum(spectrum):
        """
        Calculate optimal phase parameters using the new Minimum Entropy algorithm.
        Returns: (p0, p1) in degrees
        """
        return auto_phase_entropy(spectrum)

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
    def remove_oscillation_fft(times, amps, noise_level=None):
        """
        Removes high frequency oscillations using Butterworth Low Pass Filter.
        Auto-detects oscillation frequency using FFT.
        """
        dt = np.mean(np.diff(times))
        fs = 1.0 / dt
        n = len(amps)
        
        # 1. Analyse Spectrum to find Oscillation Frequency
        # FFT
        yf = scipy.fft.rfft(amps)
        xf = scipy.fft.rfftfreq(n, dt)
        
        # Strategy: Find dominant peak in AC component (Frequency > 0)
        # Low freq cutoff: Ignore first few bins to avoid DC/Decay components
        # Assuming decay is dominant at very low freq.
        idx_skip = max(2, int(n * 0.02)) # Skip 2% low freq or at least 2 bins
        
        if idx_skip >= len(xf):
             return {'status': "Data too short."}
             
        # Find max peak (Oscillation)
        # Search for peak in the rest of the spectrum
        idx_max = idx_skip + np.argmax(np.abs(yf[idx_skip:]))
        f_osc = xf[idx_max]
        
        # 2. Filter Design (Adaptive Moving Average)
        # A Moving Average filter over several periods of oscillation to smooth out beats.
        
        # Calculate samples per cycle
        if f_osc > 0:
            samples_per_cycle = int(round(fs / f_osc))
        else:
            samples_per_cycle = 5 
            
        # Use a larger window (e.g., 3 cycles) to crush residuals, 
        # but don't exceed ~15% of total data to avoid killing the T2 decay shape itself.
        target_cycles = 3
        window_size = samples_per_cycle * target_cycles
        
        # Constraints on window size
        max_window = int(n * 0.15) 
        window_size = min(window_size, max_window)
        window_size = max(window_size, 5)
        
        # Ensure window is odd for symmetry
        if window_size % 2 == 0:
            window_size += 1
            
        # Apply Rolling Mean with aggressive smoothing
        y_clean = pd.Series(amps).rolling(window=window_size, center=True, min_periods=1).mean().to_numpy()


        # 3. Fit T2* to the cleaned curve
        # TRIM EDGES: Filter artifacts (start/end)
        n_trim = max(5, int(n * 0.05))
        if len(times) > 2 * n_trim + 10:
            t_fit = times[n_trim:-n_trim]
            y_fit_data = y_clean[n_trim:-n_trim]
        else:
            t_fit = times
            y_fit_data = y_clean

        def exp_model(t, A, T2, C):
             return A * np.exp(-t/T2) + C
             
        # Robust Initial Guess with Bounds
        # Estimate Baseline C as the minimum of the tail
        y_min = np.min(y_fit_data)
        y_max = np.max(y_fit_data)
        
        # Estimate T2 via two-point log calculation (assuming C=y_min)
        # Slope of ln(y - C) vs t is -1/T2
        
        # Take top 10% and bottom 10% averages to be robust against noise
        n_pts = len(y_fit_data)
        n_avg = max(1, int(n_pts * 0.1))
        
        y_top = np.mean(y_fit_data[:n_avg])
        y_bot = np.mean(y_fit_data[-n_avg:])
        
        # Determine C limit based on noise_level or data properties
        # User requirement: C should be within the noise amplitude.
        if noise_level is not None and noise_level > 0:
            # If noise level is known, C cannot exceed it (or slightly above)
            C_limit = noise_level * 1.5 
        else:
            # Fallback
            C_limit = y_min + (y_max - y_min) * 0.01 
        
        # C_guess should be consistent with this limit
        C_guess = 0.0 
        
        # T2 Guess Logic
        val_start = max(1e-9, y_top - C_guess)
        val_end = max(1e-9, y_bot - C_guess)
        
        # We need local time for T2 guess
        dt_span = t_fit[-1] - t_fit[0]
        
        if dt_span > 0 and val_start > val_end:
            T2_guess = dt_span / np.log(val_start / val_end)
        else:
            T2_guess = dt_span * 10.0 # Assume very long decay if flat

        A_guess = max(1e-9, y_top - C_guess)
        
        p0 = [A_guess, T2_guess, C_guess]
        
        # Bounds: 
        # C is constrained to [0, C_limit]
        bounds = (
            [0, 0, 0], 
            [np.inf, np.inf, C_limit] 
        )
        
        fit_res = {'T2': 0, 'R2': 0, 'status': 'fit_failed'}
        try:
             # Shift time for numerical stability and A interpretation
             t_start = t_fit[0]
             
             popt, _ = curve_fit(exp_model, t_fit - t_start, y_fit_data, p0=p0, bounds=bounds, maxfev=2000)
             A_fit, T2_fit, C_fit = popt
             
             # R2 (calc on trimmed data)
             residuals = y_fit_data - exp_model(t_fit - t_start, *popt)
             ss_res = np.sum(residuals**2)
             ss_tot = np.sum((y_fit_data - np.mean(y_fit_data))**2)
             if ss_tot == 0:
                 r2 = 0
             else:
                 r2 = 1 - (ss_res / ss_tot)
             
             # Generate full length fit curve for plotting
             # exp_model now expects time relative to t_start
             y_fit_full = exp_model(times - t_start, *popt)
             
             fit_res = {
                 'status': 'success',
                 'T2': T2_fit,
                 'R2': r2,
                 'params': popt,
                 'y_fit': y_fit_full 
             }
        except Exception as e:
             fit_res['error'] = str(e)
        
        return {
            'status': 'success',
            'amps_filtered': y_clean,
            'f_osc': f_osc,
            'fit_result': {
                'status': fit_res.get('status', 'fail'),
                'T2': fit_res.get('T2', 0),
                'R2': fit_res.get('R2', 0),
                'fit_x': times,
                'fit_y': fit_res.get('y_fit', [])
            }
        }
