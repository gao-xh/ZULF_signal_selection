# Automatic Phase Correction Design Notes

**Date:** January 18, 2026
**Topic:** Auto-Phasing for ZULF NMR Data

## 1. Problem Definition
Recover the pure absorption spectrum (Lorentzian real part) from raw NMR data which suffers from phase distortions due to:
- Instrumental dead time.
- Pulse delays.
- Electronic phase shifts.

Formula: $\phi(\nu) = p_0 + p_1 \cdot \nu$

## 2. Selected Algorithm: Minimum Entropy
We will implement an **Entropy Minimization** strategy.

### Reasoning
- **Sparsity:** True absorption spectra are "sparser" (sharp peaks, near-zero baseline) compared to dispersive or phase-twisted spectra.
- **Robustness:** Works well for ZULF data which typically features distinct, sharp lines.
- **No Parameters:** Does not require pre-selecting peak positions.

### Mathematical Formulation
The objective is to minimize the entropy $E$ of the real part of the phased spectrum $S(\phi)$:

$$ E(\phi) = - \sum_i h_i \ln(h_i) $$

Where $h_i$ is the normalized absolute intensity of the real part:
$$ h_i = \frac{|Re(S_i(\phi))|}{\sum_k |Re(S_k(\phi))|} $$

*(Note: Variations like limiting the derivative or simpler sparsity metrics $\sum |Re|^\alpha$ for $\alpha < 1$ also work well).*

## 3. Implementation Strategy (`src/processing.py`)

### A. Optimization Routine
Since the landscape can have local minima (especially for $p_0$ which is periodic $2\pi$), a two-step approach is recommended:

1.  **Coarse Search ($p_0$ only)**:
    -   Fix $p_1 = 0$.
    -   Grid search $p_0$ from $0$ to $360^\circ$ (e.g., steps of $10^\circ$).
    -   Find the region of minimal entropy.

2.  **Fine Tuning (Nelder-Mead)**:
    -   Use `scipy.optimize.minimize`.
    -   Variables: $[p_0, p_1]$.
    -   Start from the best coarse $p_0$.
    -   Bounds: $p_0 \in [0, 360]$, $p_1$ usually small (correlated to dead time).

### B. Dependencies
- `scipy.optimize`: For the minimization solver.
- `numpy`: For array manipulations.

## 4. Alternatives Considered
- **Negativity Minimization** ($\sum_{Re<0} Re^2$):
    -   *Risk:* Can fail if baseline separation is poor or if real physical negatives exist.
- **Baseline Flatness**:
    -   *Risk:* Less sensitive for high-SNR ZULF data, computationally heavier to define "baseline".
