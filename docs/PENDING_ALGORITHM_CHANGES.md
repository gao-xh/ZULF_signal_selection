# Pending Algorithm Refactoring Tasks

## 1. Iterative Dynamic Peak Search (Global T2* Analysis)
- **Current Behavior**: Static search window centered at the initial peak position for all time slices.
- **Problem**: If the peak drifts or the signal is noisy, the max intensity might be picked from a neighbor peak or noise.
- **Target Behavior**:
    - **Initialization**: Lock the starting center frequency based on the user-selected peak at `Truncation Start`.
    - **Iteration**: For each subsequent time step $t$, set the search center to the peak position found at step $t-1$.
    - **Logic**: $Center_{t} = PeakPosition_{t-1}$

## 2. Search Units (Hz vs Points)
- **Current Behavior**: `search_r` is hardcoded or set as an integer value (Points).
- **Target Behavior**: 
    - Expose a `Search Width (Hz)` parameter.
    - Dynamically calculate points based on spectral resolution: `search_points = int(search_width_hz / (spectral_width / n_points))`.

## 3. UI Updates (Next Session)
- **Analysis Tab**: Add `Tracking Window (Hz)` Slider/SpinBox to the Global Map settings group.
- **Backend**: Update `BatchRelaxationWorker` to accept this new parameter and implement the stateful tracking loop.
