# Changelog

## [Unreleased] - 2026-01-16

### Fixed
- **Critical Crash**: Resolved `MemoryError` when enabling SVD on full datasets. Introduced `MAX_SVD_POINTS` safety check and moved Truncation before SVD in the pipeline.
- **SVD Defaults**: Changed default SVD state to `False` to prevent startup crashes on standard machines.
- **Data Loading**: Fixed reversed time-axis issue by adding `np.flip()` in `loader.py`, aligning behavior with the reference implementation.
- **Validator Bug**: Fixed `NameError: name 'mag' is not defined` in `src/validator.py`.
- **UI Freeze**: Moved heavy I/O and processing to background threads (`LoaderWorker`, `ProcessWorker`).

### Added
- **Peak Detection Controls**: Added UI sliders for "Min Abs Height", "Search Freq Min", and "Search Freq Max".
- **Time Domain Plot**: Added a dedicated Time Domain plot in the top-right panel for better pre-processing visualization.
- **Config**: Added new configuration constants for peak detection defaults.

### Changed
- **Workflow**: Separated "Load Data", "Refresh Processing", and "Run Analysis" into distinct steps with clearer button labels.
- **Peak Thresholding**: Changed peak detection loop to use absolute amplitude thresholds instead of relative percentage.
- **Dependencies**: Updated usage of `scipy.signal` and `find_peaks` to robustly handle empty results.
