# Notes on T2 Map Visualization Update (Traditional KDE Effect)

**Date:** 2026-03-16  
**Change:** Switch to "Traditional KDE" style visualization.  
**File:** `src/ui_main.py` -> `update_stft_t2_visuals`

## 1. Requirement
- **User Request:** "Use traditional KDE effect" and "Color bar starts from transparent".
- **Goal:** Achieve a smooth, continuous density plot that resembles `scipy.stats.gaussian_kde` output but runs efficiently on large datasets, with a visual style where low-density regions fade to transparency.

## 2. Implementation Logic

### A. Smoothing Kernel (Sigma)
- Increased `gaussian_filter` sigma from `(2, 2)` to `(3, 3)`.
- **Reason:** Traditional KDE bandwidths are usually wider than single grid bins. A larger sigma creates smoother, more organic "blobs" rather than pixelated squares, mimicking the continuous probability density function of true KDE.

### B. Transparent Colormap
- **Base:** `Turbo` (high contrast, perceptually uniform rainbow).
- **Modification:** Added an Alpha (Opactiy) gradient to the lower 15% of the colormap.
  - Value 0-15%: Alpha ramps from 0.0 (Invisible) to 1.0 (Solid).
  - Value 15-100%: Solid colors.
- **Code:**
  ```python
  turbo_colors = turbo_cmap(np.linspace(0, 1, 256))
  # Fade first 15%
  fade_len = int(256 * 0.15)
  turbo_colors[:fade_len, 3] = np.linspace(0, 1, fade_len)
  custom_cmap = LinearSegmentedColormap.from_list('turbo_transparent', turbo_colors)
  ```
- **Effect:** Background noise and zero-density regions become transparent, allowing the grid lines to show through and making the "hot spots" stand out clearly as floating density clouds.

### C. Rendering Method
- **Method:** `pcolormesh` with `shading='gouraud'`.
- **Why Gouraud?**
  - Standard `shading='flat'` or `'auto'` renders each grid bin as a solid color block (pixelated).
  - `shading='gouraud'` performs bilinear interpolation of the colors *between* grid vertices. This renders a perfectly smooth gradient, indistinguishable from a high-resolution contour plot or true KDE image, but uses the GPU/rendering engine efficiently.

## 3. Comparison
| Feature | Previous (Histogram) | New (Traditional KDE Style) |
| :--- | :--- | :--- |
| **Texture** | Blocky / Pixelated | Smooth / Continuous |
| **Smoothing** | Sigma = 2 (Sharp) | Sigma = 3 (Organic) |
| **Background** | Solid Dark Blue (Turbo min) | Transparent / Fades in |
| **Rendering** | Flat Shading | Gouraud Interpolation |

This approach satisfies the "Traditional KDE" visual requirement while maintaining the $O(N)$ performance of the histogram method.
