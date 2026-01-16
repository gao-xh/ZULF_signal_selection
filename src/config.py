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

# Candidate Search: How sensitive is the initial peak picking on the N_max spectrum?
# Value is relative to the maximum peak height (0.1 = 10% of max)
PEAK_DETECTION_THRESHOLD = 0.1

# Back-tracing: How many points +/- the center to search/integrate when extracting intensity
# Larger window = more robust against frequency drift, but risk of overlapping peaks
PEAK_SEARCH_WINDOW = 5

# Default thresholds for classifying a signal as "Good"
# These are the initial values for the UI sliders
DEFAULT_R2_THRESHOLD = 0.8
DEFAULT_SLOPE_THRESHOLD = 0.1


# --- 4. Processing Defaults ---
# Default parameters for the signal processing pipeline
DEFAULT_SAVGOL_WINDOW = 31
DEFAULT_SAVGOL_ORDER = 3
DEFAULT_APOD_T2STAR = 0.05  # Decay rate for exponential apodization
DEFAULT_SVD_RANK = 5
DEFAULT_ENABLE_SVD = True


# --- 5. UI Configuration ---
UI_WINDOW_TITLE = "ZULF Signal Selection - Progressive Validator"
UI_WINDOW_SIZE = (1200, 800)

# Parameter Ranges for UI Sliders: (Min, Max, Step, Default)
# Format: 'key': (min, max, step, default)
UI_PARAM_RANGES = {
    'savgol_window': (1, 101, 2, DEFAULT_SAVGOL_WINDOW),
    'savgol_order': (1, 6, 1, DEFAULT_SAVGOL_ORDER),
    'apod_t2star': (0.01, 1.0, 0.01, DEFAULT_APOD_T2STAR),
    'phase_0': (-180, 180, 1, 0.0),
    'min_r2': (0.0, 1.0, 0.01, DEFAULT_R2_THRESHOLD),
    'min_slope': (-0.5, 2.0, 0.1, DEFAULT_SLOPE_THRESHOLD)
}
