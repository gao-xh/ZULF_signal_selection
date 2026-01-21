# Changelog

## [Current] - 2026-XX-XX
### Added
- **Iterative Peak Tracking**: Implemented dynamic peak search ($Center_t = Peak_{t-1}$) in Global Analysis.
- **Analysis UI**: Added "Enable Iterative Tracking" and "Tracking Window (Hz)" inputs.
- **UI Refactor**: Split interface into Processing, Detection, and Analysis tabs.
- **Persistence**: Updated settings to save/load tracking parameters.


## [Unreleased] - 2026-01-16

### Fixed
- **Critical Crash**: Resolved `MemoryError` when enabling SVD on full datasets. Introduced `MAX_SVD_POINTS` safety check and moved Truncation before SVD in the pipeline.
- **SVD Defaults**: Changed default SVD state to `False` to prevent startup crashes on standard machines.
- **Data Loading**: Fixed reversed time-axis issue by adding `np.flip()` in `loader.py`, aligning behavior with the reference implementation.
- **Validator Bug**: Fixed `NameError: name 'mag' is not defined` in `src/validator.py`.
- **UI Freeze**: Moved heavy I/O and processing to background threads (`LoaderWorker`, `ProcessWorker`).

### Added
- **Noise Estimation**: Implemented selectable noise calculation methods (Global Region vs Local Window) to handle different baseline conditions.
- **Visualization**: Added real-time visual feedback (overlay lines and regions) on the spectrum plot for Search Range, Noise Region, and Amplitude Thresholds.
- **Peak Detection Controls**: Added UI sliders for "Min Abs Height", "Search Freq Min", and "Search Freq Max".
- **Time Domain Plot**: Added a dedicated Time Domain plot in the top-right panel for better pre-processing visualization.
- **Config**: Added new configuration constants for peak detection defaults.

### Changed
- **UI UX**: Implemented dynamic range limits for sliders based on loaded data dimensions (max frequency, max amplitude).
- **Workflow**: Separated "Load Data", "Refresh Processing", and "Run Analysis" into distinct steps with clearer button labels.
- **Peak Thresholding**: Changed peak detection loop to use absolute amplitude thresholds instead of relative percentage.
- **Dependencies**: Updated usage of `scipy.signal` and `find_peaks` to robustly handle empty results.
