from pathlib import Path

"""
Global Configuration for ZULF Signal Selection
==============================================
This file centralizes all configurable parameters, paths, and default values.
Modify this file to adjust the behavior of the application without changing core logic.
"""

# --- 1. Paths & Environment ---
# The absolute path to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Path to the reference library (Data-Process)
# If you move the references folder, update this path.
REFERENCES_DIR = PROJECT_ROOT / 'references' / 'Data-Process'


# --- 2. Data Loading ---
# Default sampling rate if 0.ini is missing or unreadable
DEFAULT_SAMPLING_RATE = 6000.0

# Pattern to identify scan files
SCAN_FILE_PATTERN = "*.dat"


# --- 3. Validation Logic (The Algorithm) ---
# How many checkpoints (subsets of averages) to generate for the regression curve
# More points = smoother curve but slower processing
CHECKPOINT_COUNT = 20

# Candidate Search: Absolute Minimum Intensity Threshold
# Peaks below this absolute value will be ignored
DEFAULT_PEAK_HEIGHT = 1000.0
DEFAULT_SEARCH_FREQ_MIN = 0.0
DEFAULT_SEARCH_FREQ_MAX = 1000.0

# Noise Calculation: Frequency range to use for noise estimation
DEFAULT_NOISE_FREQ_MIN = 300.0
DEFAULT_NOISE_FREQ_MAX = 400.0
DEFAULT_NOISE_METHOD = 'global' # 'global' or 'local'
DEFAULT_LOCAL_NOISE_WINDOW = 200 # +/- points around peak

# Back-tracing: How many points +/- the center to search/integrate when extracting intensity
# Larger window = more robust against frequency drift, but risk of overlapping peaks
PEAK_SEARCH_WINDOW = 5

# Default thresholds for classifying a signal as "Good"
# These are the initial values for the UI sliders
DEFAULT_R2_THRESHOLD = 0.8
DEFAULT_SLOPE_THRESHOLD = 0.1


# --- 4. Processing Defaults ---
# Default parameters for the signal processing pipeline
DEFAULT_SAVGOL_WINDOW = 301 # Must be odd, reference uses 300~ but usually needs optimization
DEFAULT_SAVGOL_ORDER = 2
DEFAULT_APOD_T2STAR = 0.05  # Decay rate for exponential apodization
DEFAULT_SVD_RANK = 5
DEFAULT_ENABLE_SVD = True


# --- 5. UI Configuration ---
UI_WINDOW_TITLE = "ZULF Signal Selection - Progressive Validator"
UI_WINDOW_SIZE = (1200, 800)

# Parameter Ranges for UI Sliders: (Min, Max, Step, Default)
# Format: 'key': (min, max, step, default)
UI_PARAM_RANGES = {
    'savgol_window': (3, 12000, 1, DEFAULT_SAVGOL_WINDOW), # Reference range 2-12000
    'savgol_order': (1, 20, 1, DEFAULT_SAVGOL_ORDER), # Reference range 1-20
    'apod_t2star': (0.0001, 10.0, 0.001, DEFAULT_APOD_T2STAR),
    'phase_0': (-360, 360, 1, 0.0),
    'phase_1': (-50000, 50000, 100, 0.0), # Phase 1 (Linear Phase) usually needs large range for us/ms delays
    'trunc_start': (0, 3000, 1, 0), # Enhanced range matching reference
    'trunc_end': (0, 60000, 10, 0), # Reference range is 0-60000
    'peak_height': (0.0, 10000000.0, 100.0, DEFAULT_PEAK_HEIGHT),
    'search_freq_min': (0.0, 2000.0, 1.0, DEFAULT_SEARCH_FREQ_MIN),
    'search_freq_max': (0.0, 2000.0, 1.0, DEFAULT_SEARCH_FREQ_MAX),
    'peak_window': (1, 20, 1, PEAK_SEARCH_WINDOW),
    'min_r2': (0.0, 1.0, 0.01, DEFAULT_R2_THRESHOLD),
    'noise_freq_min': (0.0, 2000.0, 1.0, DEFAULT_NOISE_FREQ_MIN),
    'noise_freq_max': (0.0, 2000.0, 1.0, DEFAULT_NOISE_FREQ_MAX),
    'local_noise_window': (10, 1000, 10, DEFAULT_LOCAL_NOISE_WINDOW),
    'min_slope': (-0.5, 2.0, 0.001, DEFAULT_SLOPE_THRESHOLD),
    
    # Baseline
    'baseline_lambda': (1.0, 1000000.0, 10.0, 1000.0),
    'baseline_p': (0.0001, 1.0, 0.001, 0.01),
    'baseline_niter': (1, 100, 1, 10)
}

