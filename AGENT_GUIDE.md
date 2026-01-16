# 📘 ZULF Signal Selection - Agent Guide & Developer Handover

> **Note**: This guide serves as the "source of truth" for AI agents and developers working on this project.

### 1. Project Overview
**Project Name**: ZULF Signal Selection (ML_ZULF)
**Goal**: Identify valid signal peaks from experimental data by verifying that peak intensity enhances as data volume (averages) increases. This validates signals based on the principle that Signal-to-Noise Ratio (SNR) is proportional to the square root of the number of scans.

*   **Core Context**: Low-field NMR signals are often buried in noise. We leverage the $\text{SNR} \propto \sqrt{N}$ scaling law.
    *   **Signal**: Coherent integration leads to intensity growth proportional to $\sqrt{N}$ (relative to noise).
    *   **Noise**: Random phases and amplitudes cause it to average out (cancelation) or fluctuate randomly without a growth trend.
    *   **Method**: Peaks that grow consistently with more averages are treated as signals; those that do not are noise.
*   **Workspace Structure**:
    *   `src/`: Core logic and helper functions.
    *   `notebooks/`: Exploratory Data Analysis (EDA) and model prototyping.
    *   `tests/`: Unit and integration tests.
    *   `references/Data-Process/`: The reference codebase for NMR processing logic.

### 2. Development Standards
*   **Language**: Python 3.x
*   **Code Comments**: **MUST BE IN ENGLISH**. No Chinese characters in the source code files (to avoid encoding issues and maintain standard).
*   **UI/Interaction Design**: For any data processing parameter adjustment (e.g., ranges, thresholds), **MUST** use the "Slider + SpinBox" synchronized pattern found in the reference code. This provides both intuitive sliding and precise numerical input.
*   **Docstrings**: Use Google or NumPy style docstrings.
*   **Type Hinting**: Strongly encouraged for all public functions in `src/`.

### 3. Architecture & Key Logic (Planned)

#### A. Data Pipeline
1.  **Ingestion**: Load raw ZULF data (likely time-domain FID).
2.  **Preprocessing**: Apply basic text/filtering.
    *   *Reference*: Use logic from `references/Data-Process/nmr_processing_lib`.
3.  **Feature Extraction**: Extract features relevant to peak evolution (e.g., peak height vs. scan count, consistency of frequency).
4.  **Signal Validation Strategy**:
    *   **Logic**: Analyze peak evolution across cumulative averages (or progressive scan blocks).
    *   **Criterion**: Valid signals must show growth/enhancement consistent with $\sqrt{N}$ scaling (SNR vs Scans), whereas noise should average out or behave randomly.
    *   **Classification**: Distinguish "Signal" (growth) vs "Noise" (random/decay) based on this trend.

#### B. Reference Utilization
*   The `references/Data-Process` folder contains robust implementations for:
    *   SVD Denoising (`nmr_processing_lib.processing.filtering`)
    *   FFT & Phasing
    *   Data loading utilities
*   **Strategy**: Port or reference these utilities into `src/` rather than rewriting them from scratch.

### 4. Implementation Strategy (Progressive Validation)

#### A. Data Structure & incremental Loading
*   **Configuration**: Use `src/config.py` for all global constants (paths, defaults, thresholds, UI ranges). Do not hardcode values in logic files.
*   **Existing Format**: The reference system uses individual `.dat` files for each scan (e.g., `0.dat`, `1.dat`...) in a folder.
*   **Optimization**: Instead of reloading files for each checkpoint, use **Cumulative Sum** (Running Sum).
    *   `Sum_N = Sum_{N-1} + Scan_N`
    *   `Average_N = Sum_N / N`
    *   This avoids I/O bottlenecks when computing averages for N=10, 100, 1000, etc.

3.  **Unified Preprocessing**:
    *   **Phase Sync**: All subsets must use the **same phase correction parameters** derived from the full dataset ($N_{max}$). Do not auto-phase each subset individually to avoid jitter.
    *   **Pipeline**: Load Raw $\to$ Pre-processing (Savgol/Filter) $\to$ Average $\to$ FFT $\to$ Phase Correction (Fixed).

#### B. Validation Workflow
1.  **Dynamic Checkpoints**: Generate checkpoints $(N_i)$ based on total scan count $M$ to ensure uniform distribution on the $\sqrt{N}$ scale.
    *   Formula: $N_i \approx (\frac{i}{K}\sqrt{M})^2$
2.  **Sampling Modes**:
    *   **Mode A (Sequential)**: Cumulative average $[0, N_i]$. Fast, sensitive to drift.
    *   **Mode B (Bootstrap/Resampling)**: For a fixed $N_i$, randomly sample $N_i$ scans $K$ times (e.g., 5 times).
        *   Result: Mean SNR $\pm$ StdDev.
        *   Benefit: Provides error bars and checks statistical stability.
3.  **Trace Extraction**: For each candidate frequency peak, extract its intensity at every checkpoint.
4.  **Regression Analysis**:
    *   Fit the intensity $I$ against $\sqrt{N}$.
    *   Calculate **Correlation Coefficient ($R^2$)** and **Slope**.
    *   **Signal**: High $R^2$, Positive Slope (Consistent with $\sqrt{N}$ growth).
    *   **Noise**: Low $R^2$, or Flat/Negative Slope (Random phases cancel out or fluctuate unpredictably).

### 5. Final Output & Visualization
*   **Macro View (Traffic Light Spectrum)**: A full spectrum view where peaks are color-coded (Green=Signal, Red=Noise, Yellow=Unsure).
*   **Micro View (Evolution Plot)**: Click on any peak to see its "Growth Curve" ($Intensity$ vs $\sqrt{N}$).
    *   This is the definitive proof for signal validity.
*   **Control Panel**: Use "Slider + SpinBox" to adjust decision thresholds (e.g., Min $R^2$, Min Slope) in real-time.

### 6. Critical Watchlist (Avoid these pitfalls)

1.  **Path dependencies**:
    *   Never hardcode absolute paths (e.g., `C:\Users\...`). Use `pathlib` and relative paths.
    *   Data files should reside in `data/` and be ignored by Git if they are large.

2.  **Reference Code Isolation**:
    *   Do not modify files inside `references/Data-Process` directly if possible. Copy useful functions to `src/` or import them if the path is added to `sys.path` (Copying is often safer for decoupling).

3.  **Reproducibility**:
    *   Notebooks (`notebooks/`) should be reproducible. Move stable logic from notebooks to `src/` modules frequently.

### 6. Next Steps
*   [ ] Implement `src/loader.py` with `ProgressiveLoader` class.
*   [ ] Create `src/validator.py` with regression logic.
*   [ ] Port key processing functions (FFT, Phase) to `src/processing.py`.
*   [ ] Create a demo notebook to visualize Signal vs Noise evolution.
