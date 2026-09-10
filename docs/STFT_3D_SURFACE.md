# STFT 3D surface with time-slice spectra

## Independent axis zoom

The surface appearance uses the original semi-transparent rendering (default opacity 0.85), before the overlap-color experiments. Those experiments remain in their GitHub backup branches.

The X, Y and Z rows each provide editable Min/Max bounds, a synchronized zoom slider/numeric field and Reset. Zoom is centered on that axis's current range. Editing one axis preserves the other two ranges and the camera orientation. Choose X, Y or Z under **Mouse wheel zoom**, then scroll over the plot to zoom that axis only.

**Reset all axes** restores data-fit ranges without changing the camera. **Reset 3D view** restores camera orientation without resetting axis ranges. Changing opacity or slice settings preserves current ranges. Switching linear/dB mode resets only Z, since its units change. Loading/refreshing STFT data initializes all axes for the new result. Invalid Min/Max edits restore the previous valid range. Plot navigation changes update the numeric fields.

These controls change displayed data ranges, not STFT values or the physical lengths of the axis box. The color scale remains tied to the plotted surface rather than being normalized to the selected Z range.

Axis-zoom backups: `backup/pre-axis-zoom-20260910` and `backup/post-axis-zoom-20260910`. Development branch: `feature/stft3d-axis-zoom-20260910`.

Open the existing application, load data and run processing. In the Spectrogram controls, set the STFT window, frequency folding and visible frequency range, then click **Show 3D Surface**.

An independent, resizable window displays a continuous surface:

- X: Time (s), including the existing truncation offset.
- Y: Frequency (Hz), or absolute frequency when the existing folding option is enabled.
- Z: STFT magnitude in input units, optionally amplitude in dB relative to one input unit.

Each dark line is a spectrum at a fixed time: X is constant, Y runs across frequencies, and Z is the magnitude at that time. The time-slice control selects evenly distributed STFT frame indices, including the endpoints when more than one line is requested. The number of lines cannot exceed the available frames. Surface opacity and line visibility are adjustable. Drag to rotate, use the independent navigation toolbar to navigate/export, and select Reset 3D View to restore the initial orientation.

No per-frame normalization, vertical offsets or new smoothing are applied. The view snapshots the existing STFT result; it does not change analysis arrays, fitting or the original heatmap. Closing the window preserves it for reopening. New processed data clears the surface until refreshed. Updating the spectrogram refreshes an already-visible surface using the current frequency range/folding.

The 3D window has its own amplitude/dB switch, independent of the heatmap display option. The existing STFT computation, including its window, overlap and boundary padding behavior, is unchanged. Boundary frames can contain padded samples and should not be interpreted as complete instantaneous spectra.

For large datasets, the surface mesh uses a limited display grid; the caption reports sampling. Slice lines retain all visible frequency bins. Original analysis arrays stay at full resolution. Mesh sampling can omit narrow features between sampled bins; use the slice lines and original heatmap for detailed inspection. Opaque surfaces may occlude rear lines; rotate or lower opacity.

Verification uses synthetic data: exact time/frequency slice coordinates, unnormalized amplitudes, dB values, camera preservation, invalid inputs and the existing main-window STFT workflow. No real experimental dataset was used for acceptance.

Before-change backup: `backup/pre-stft3d-20260909` in `gao-xh/ZULF_signal_selection`.
Implementation branch: `feature/stft3d-surface-20260909`.
After-change backup: `backup/post-stft3d-20260909`.
