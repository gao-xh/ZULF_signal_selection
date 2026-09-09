# Algorithm Note: T2* Map Visualization (KDE-like)

**Date:** 2026-03-16  
**Context:** Optimization of the "Global Map" in the "T2 Analysis" tab.  
**Implementation:** `src/ui_main.py` -> `update_stft_t2_visuals`

## 1. Background
The goal is to visualize the relationship between Frequency (Hz) and T2* Decay Time (s). 
- **Previous Approach:** Scatter plots or basic Delaunay triangulation (`griddata`).
- **Issues:**
  - Scatter plots suffer from overplotting when thousands of peaks are detected.
  - Triangulation creates jagged artifacts and fills empty space inappropriately.
  - Hard to distinguish "dense" signal regions from noise outliers.

## 2. Current Algorithm: Weighted Binned KDE
Instead of true Kernel Density Estimation (which is computationally expensive for $N > 10^4$ points), we use a **Weighted 2D Histogram with Gaussian Smoothing**. This effectively approximates a KDE but is orders of magnitude faster.

### Step 1: Weight Calculation
We calculate a weight $W$ for every detected peak. The goal is to highlight signals that are both *strong* (High Amplitude) and *reliable* (High $R^2$ fit).

$$ W_i = A_i \times (R^2_i)^4 $$

- **$A_i$**: Amplitude of the spectral peak.
- **$R^2_i$**: Goodness of fit (0 to 1).
- **Power of 4**: The $R^2$ term is raised to the 4th power to heavily penalize poor fits. A fit with $R^2=0.9$ contributes only $65\%$ as much as a perfect fit, while $R^2=0.5$ contributes effectively nothing ($6\%$).

### Step 2: Adaptive Binning
We define a 400x400 grid.
- **X-Axis (Frequency)**: Linearly spaced bins from `min(freq)` to `max(freq)`.
- **Y-Axis (T2 Time)**: **Logarithmically** spaced bins from `min(t2)` to `max(t2)`.
  - usage: `np.logspace(...)`
  - Reason: T2 values span orders of magnitude (e.g., 0.1s to 10s). Linear binning would crowd all short T2s into one bin.

### Step 3: Histogram Accumulation
We use `numpy.histogram2d` to sum the weights $W_i$ into the grid cells.

```python
H, xedges, yedges = np.histogram2d(
    x=frequencies, 
    y=t2_values, 
    bins=[x_grid, y_grid_log], 
    weights=weights
)
```

### Step 4: Gaussian Smoothing
To convert the raw histogram (discrete counts) into a smooth probability density map, we apply a Gaussian filter.

```python
from scipy.ndimage import gaussian_filter
H_smooth = gaussian_filter(H.T, sigma=(2, 2))
```
- **Sigma=2**: The smoothing kernel has a standard deviation of 2 bins. This smears the signal slightly to connect adjacent peaks and create a "heat" effect.

### Step 5: Rendering
The result is plotted using `pcolormesh` with the `turbo` colormap, which provides high perceptual contrast.

## 3. Advantages
1.  **Speed**: $O(N)$ complexity for binning, extremely fast compared to $O(N^2)$ for true KDE.
2.  **Clarity**: High-density regions naturally "glow".
3.  **Robustness**: The $R^2$ weighting automatically hides noise (low $R^2$) without needing strict hard thresholds.
4.  **Log-Scale Support**: Handles the wide dynamic range of T2 relaxation times correctly.
