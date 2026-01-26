import numpy as np
from scipy.optimize import minimize

def apply_phase_correction(spectrum, phi0, phi1, pivot_index=None):
    """
    Apply zero-order and first-order phase correction with pivot support.
    
    Args:
        spectrum (np.ndarray): Complex spectrum.
        phi0 (float): Zero-order phase in degrees.
        phi1 (float): First-order phase in degrees (total phase shift across bandwidth).
        pivot_index (int): Index of the pivot point. If None, defaults to center.
        
    Returns:
        np.ndarray: Phase-corrected complex spectrum.
    """
    N = len(spectrum)
    
    if pivot_index is None:
        pivot_index = N // 2
    
    # Convert to radians
    phi0_rad = np.deg2rad(phi0)
    phi1_rad = np.deg2rad(phi1)
    
    # Create normalized frequency axis centered at pivot
    # Range: -pivot/N to (N-pivot)/N
    freq_norm = (np.arange(N) - pivot_index) / N
    
    # Total phase correction
    # phi(f) = phi0 + phi1 * (f - f_pivot)
    phase_corr = phi0_rad + phi1_rad * freq_norm
    
    return spectrum * np.exp(1j * phase_corr)

def minimum_entropy_cost(params, spectrum):
    """
    Calculate the entropy of the real part of the spectrum.
    
    Equation: E = - sum( h_i * ln(h_i) )
    Where h_i = |Re(Si)| / sum(|Re(S)|)
    
    Args:
        params: list/tuple of [phi0, phi1] or [phi0]
        spectrum: complex spectrum array
    
    Returns:
        float: Entropy value (to be minimized)
    """
    if len(params) == 2:
        p0, p1 = params
    else:
        p0, p1 = params[0], 0.0
        
    # Apply phase
    phased = apply_phase_correction(spectrum, p0, p1)
    real_spec = np.real(phased)
    
    # Calculate probability distribution h_i
    # Using absolute value of real part ensures h_i >= 0
    # Ideally, for a purely absorptive spectrum, Re should be mostly positive
    # But we use abs here to handle both positive and negative peaks if they exist,
    # or strictly positive if that's the prior knowledge. 
    # The design guide says: h_i = |Re(Si)| / Sum(|Re|)
    
    abs_real = np.abs(real_spec)
    total_intensity = np.sum(abs_real)
    
    if total_intensity == 0:
        return 0.0
    
    h = abs_real / total_intensity
    
    # Avoid log(0) by adding a tiny epsilon strictly for the log calculation
    epsilon = 1e-12
    
    # E = - sum( h * ln(h) )
    # Note: h is between 0 and 1. ln(h) is negative. Entropy is positive.
    # Elements with h=0 contribute 0 to entropy (lim x->0 of x*ln(x) is 0)
    
    # Vectorized computation
    # Mask out zeros to avoid warnings
    mask = h > epsilon
    h_valid = h[mask]
    
    entropy = -np.sum(h_valid * np.log(h_valid))
    
    # Add a penalty for negative values if we strictly expect positive peaks?
    # The design doc mentions "metric variations like... limiting the derivative or simpler sparsity metrics".
    # But strictly speaking, Min Entropy is just the above.
    # However, sometimes a "penalty term" helps p0 convergence. 
    # Let's stick to the pure definition first as per guide.
    
    return entropy

def auto_phase_entropy(spectrum):
    """
    Two-step Entropy Minimization for Auto-Phasing.
    
    1. Coarse search for p0 (0..360) with p1=0.
    2. Fine optimization for (p0, p1) using Nelder-Mead.
    
    Returns:
        tuple: (best_p0, best_p1) in degrees
    """
    # 1. Coarse Search (p0 only)
    step = 10
    search_grid = np.arange(0, 360, step)
    best_p0_coarse = 0
    min_entropy = float('inf')
    
    for p0 in search_grid:
        val = minimum_entropy_cost([p0, 0], spectrum)
        if val < min_entropy:
            min_entropy = val
            best_p0_coarse = p0
            
    # 2. Fine Tuning (Nelder-Mead)
    # Start from the best coarse p0, and p1=0
    x0 = [best_p0_coarse, 0.0]
    
    # Wrapper for minimize that handles arguments
    # minimize passes 'x' as first arg to fun
    res = minimize(
        fun=minimum_entropy_cost, 
        x0=x0, 
        args=(spectrum,),
        method='Nelder-Mead',
        tol=1e-4
    )
    
    final_p0, final_p1 = res.x
    
    # Normalize p0 to [0, 360)
    final_p0 = final_p0 % 360
    
    return final_p0, final_p1
