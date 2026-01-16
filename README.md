# ZULF Signal Selection - Progressive Validator

A specialized tool for Zero-to-Ultra-Low-Field (ZULF) NMR signal analysis. This application helps distinguish true signals from noise by analyzing signal evolution across cumulative averages ($\sqrt{N}$ scaling).

## Key Features

### 1. Progressive Signal Validation
- **Evolution Analysis**: Tracks signal intensity vs. $\sqrt{N}$ (scan count).
- **Automated Verdict**: Uses Linear Regression to classify peaks as "Signal" (Linear Growth) or "Noise" (Random Walk).
- **Traffic Light System**: Green (Confirmed Signal) / Red (Noise) markers on the spectrum.

### 2. Advanced Signal Processing
- **SVD Denoising (Cadzow)**: Optional denoising step (Note: memory intensive, disabled by default).
- **Data Truncation**: Flexible start/end truncation to remove switching noise or pulse artifacts.
- **Savgol Filter**: Baseline correction in time domain.
- **Apodization**: Exponential windowing ($T_2^*$) support.

### 3. Interactive UI
- **Noise Estimation**: Switch between **Global Region** (fixed frequency band) or **Local Window** (around peak) for flexible SNR calculation.
- **Visual Feedback**: Real-time plotting with overlay visualization of Search Ranges, Noise Regions, and Thresholds.
- **Param Tuning**: Slider-based control with dynamic range adaptation.

## Recent Updates (2026-01-16)

### Core Logic Improvements
- **Noise Calculation**: Added dual-mode noise estimation (Global/Local) to handle complex baselines.
- **Corrected Time-Domain Loading**: Fixed data loading direction to align with reference implementation (`np.flip`).
- **Peak Detection**:
    - Switched from relative percentage to **Absolute Intensity Threshold** for precise noise rejection.
    - Added **Frequency Range Filtering** (Min/Max Hz) to focus analysis on specific bands.
- **Stability**:
    - Fixed `MemoryError` issues with SVD on large datasets by enforcing data limits and optimizing pipeline order.
    - Fixed `NameError` in Validation Worker.

### UI Enhancements
- Added new "Peak Detection" settings group in the main panel.
- Unified "Min Height" to absolute units matching the plot amplitude.
- Refined button labels for clarity ("Load Data" -> "Refresh Processing" -> "Run Progressive Analysis").
- Added Time Domain visualization widget.

## Usage

1. **Load Data**: Select folders containing `.dat` files.
2. **Adjust Processing**: Set Phase, Windowing, and Truncation to optimize the `Time Domain Signal`.
3. **Configure Detection**:
    - Set **Search Freq Min/Max** to your region of interest.
    - Adjust **Min Abs Height** to sit just above the noise floor.
4. **Run Analysis**: Click "Run Progressive Analysis" to perform the evolutionary check.
5. **Inspect**: Click on any marked peak to see its regression curve in the bottom panel.

## Requirements
- Python 3.9+
- NumPy, SciPy, Matplotlib, Pandas, PySide6

## Credits
Project initialized from https://github.com/gao-xh/ZULF_signal_selection
