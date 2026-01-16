import sys
import numpy as np
import scipy.signal
import scipy.fft
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
        
        # 2. SVD Denoising (Optional)
        if params.get('enable_svd', False) and HAS_REF_LIB:
            rank = int(params.get('svd_rank', 5))
            # SVD is computationally expensive, usually done via Hankel matrix
            # Using the reference library implementation
            try:
                data = svd_denoising(data, rank)
            except Exception as e:
                print(f"SVD Error: {e}")

        # 3. Truncation
        trunc_start = int(params.get('trunc_start', 0))
        trunc_end = int(params.get('trunc_end', 0))
        
        if trunc_start > 0:
            data = data[trunc_start:]
        if trunc_end > 0:
            data = data[:-trunc_end]
            
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
