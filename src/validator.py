import numpy as np
from scipy.signal import find_peaks
from scipy.stats import linregress
import pandas as pd

class SignalValidator:
    """
    Core Logic for Signal Selection based on SNR vs sqrt(N) Evolution.
    """
    
    def __init__(self, loader, processor):
        self.loader = loader
        self.processor = processor
        self.evolution_data = {} # Stores traces for each candidate
        self.results = [] # Stores regression results

    def run_validation(self, processing_params, checkpoints=None, detection_threshold=0.1, peak_window=5):
        """
        Main execution flow.
        
        Args:
            processing_params (dict): Parameters for SVD, Apodization, etc.
            checkpoints (list): List of N to check. If None, auto-generated.
            detection_threshold (float): Relative threshold for picking candidates from N_max spectrum.
            peak_window (int): Points +/- center to search/integrate for peak intensity.
            
        Returns:
            pd.DataFrame: Summary of findings.
        """
        
        # 1. Setup Checkpoints
        if checkpoints is None:
            checkpoints = self.loader.generate_checkpoints(num_points=20)
            
        if not checkpoints:
            raise ValueError("No valid checkpoints generated. Check data folder.")
            
        max_scan = checkpoints[-1]
        
        # 2. Phase 0: Golden Reference Generation (N_max)
        # We need to process the full dataset first to:
        #   a) Determine Auto-Phase parameters
        #   b) Find Candidate Peaks
        
        print(f"Generating Golden Reference (N={max_scan})...")
        # Load all data for N_max (using stream to get just the last one effectively, 
        # or load_all if memory allows. Let's use stream to be consistent)
        
        golden_fid = None
        # We iterate to the end
        for n, data in self.loader.stream_process([max_scan]):
            golden_fid = data
            
        if golden_fid is None:
            raise ValueError("Failed to load golden reference data.")
            
        # Process Golden Spectrum
        # First, run initially to get spectrum for auto-phasing
        freqs, raw_spec = self.processor.process_fid(golden_fid, processing_params, self.loader.sampling_rate)
        
        # Auto Phase (if requested or not set)
        if processing_params.get('phase_mode') == 'auto':
            p0, p1 = self.processor.auto_phase_spectrum(raw_spec)
            processing_params['p0'] = p0
            processing_params['p1'] = p1
            print(f"Auto-Phase Locked: p0={p0:.2f}, p1={p1:.2f}")
            
            # Reprocess with locked phase
            freqs, golden_spec = self.processor.process_fid(golden_fid, processing_params, self.loader.sampling_rate)
        else:
            golden_spec = raw_spec

        # 3. Candidate Search
        # Magnitude spectrum for picking
        mag_spec = np.abs(golden_spec)
        # Dynamic threshold: e.g., 3 * median_noise or relative to max
        # Simple relative threshold for now
        height_thr = np.max(mag_spec) * detection_threshold
        
        peaks, _ = find_peaks(mag_spec, height=height_thr, distance=10)
        candidate_indices = peaks
        candidate_freqs = freqs[peaks]
        
        print(f"Found {len(candidate_indices)} candidates at: {candidate_freqs}")
        
        # Initialize storage
        # Key: freq_index
        self.evolution_data = {idx: {'N': [], 'sqrt_N': [], 'Intensity': [], 'Noise': [], 'SNR': []} 
                              for idx in candidate_indices}
        
        # 4. Back-Tracing / Stream Processing
        print("Starting Back-Tracing Evolution Loop...")
        for n, fid in self.loader.stream_process(checkpoints):
            # Process with LOCKED parameters
            _, spec = self.processor.process_fid(fid, processing_params, self.loader.sampling_rate)
            mag = np.abs(spec)
            
            for idx in candidate_indices:
                # Local Window Search Strategy (Anti-Drift)
                # Search +/- peak_window points
                start = max(0, idx - peak_window)
                end = min(len(mag), idx + peak_window + 1)
                window_slice = mag[start:end]
                
                # Metric: Max Intensity in Window
                intensity = np.max(window_slice)
                
                # Metric: Local Noise
                # Look further away: e.g. idx +/- (20 to 50)
                noise_window_start = max(0, idx - 50)
                noise_window_end = min(len(mag), idx + 50)
                
                # Exclude the peak region itself
                mask = np.ones(noise_window_end - noise_window_start, dtype=bool)
                # Mask out center 20 points
                center_local = idx - noise_window_start
                mask_start = max(0, center_local - 10)
                mask_end = min(len(mask), center_local + 10)
                mask[mask_start:mask_end] = False
                
                noise_region = mag[noise_window_start:noise_window_end][mask]
                
                if len(noise_region) > 0:
                    noise_lvl = np.std(noise_region)
                else:
                    noise_lvl = 1.0 # Safe fallback
                    
                snr = intensity / noise_lvl if noise_lvl > 0 else 0
                
                # Record
                self.evolution_data[idx]['N'].append(n)
                self.evolution_data[idx]['sqrt_N'].append(np.sqrt(n))
                self.evolution_data[idx]['Intensity'].append(intensity)
                self.evolution_data[idx]['Noise'].append(noise_lvl)
                self.evolution_data[idx]['SNR'].append(snr)

        # 5. Regression Analysis
        summary = []
        for idx in candidate_indices:
            data = self.evolution_data[idx]
            x = data['sqrt_N']
            y = data['SNR']
            
            if len(x) > 2:
                slope, intercept, r_value, p_value, std_err = linregress(x, y)
                r_squared = r_value ** 2
            else:
                slope, r_squared = 0, 0
                
            freq = freqs[idx]
            
            # Simple Verdict Logic
            # Can be refined later or adjusted in UI
            is_signal = (r_squared > 0.8) and (slope > 0.1)
            
            summary.append({
                'Freq_Hz': freq,
                'Index': idx,
                'Ref_SNR': y[-1], # SNR at N_max
                'Slope': slope,
                'R2': r_squared,
                'Verdict': 'Signal' if is_signal else 'Noise'
            })
            
        self.results = pd.DataFrame(summary)
        return self.results, freqs, golden_spec, self.evolution_data

