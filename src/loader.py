import os
import struct
import numpy as np
from pathlib import Path
from src.config import SCAN_FILE_PATTERN, DEFAULT_SAMPLING_RATE

class ProgressiveLoader:
    """
    Handles loading of NMR data in a progressive manner (cumulative averaging).
    Now supports multiple folders combined sequentially.
    """
    
    def __init__(self, folder_paths):
        """
        Initialize the loader with one or more experiment folder paths.
        
        Args:
            folder_paths (str, Path, or list): Path(s) to folder(s) containing .dat files.
        """
        if isinstance(folder_paths, (str, Path)):
            self.folder_paths = [Path(folder_paths).resolve()]
        else:
            self.folder_paths = [Path(p).resolve() for p in folder_paths]

        self.scan_files = []
        
        # Aggregate files from all folders in order
        for folder in self.folder_paths:
            if not folder.exists():
                print(f"Warning: Folder not found: {folder}")
                continue
                
            # Discover and sort files in this folder
            # Using config for pattern
            folder_files = list(folder.glob(SCAN_FILE_PATTERN))
            valid_files = [f for f in folder_files if f.stem.isdigit()]
            valid_files.sort(key=lambda f: int(f.stem))
            
            self.scan_files.extend(valid_files)
        
        if not self.scan_files:
            raise FileNotFoundError("No valid .dat files found in provided folders.")
            
        self.total_scans = len(self.scan_files)
        # Read sampling rate from the specific folder of the first file, or first valid folder
        self._primary_folder = self.folder_paths[0] if self.folder_paths else Path(".")
        self.sampling_rate = self._read_sampling_rate()
        
    def _read_sampling_rate(self):
        """Read sampling rate from 0.ini of the first folder or default from config."""
        sampling_rate = DEFAULT_SAMPLING_RATE
        for folder in self.folder_paths:
            ini_file = folder / '0.ini'
            if ini_file.exists():
                try:
                    with open(ini_file, 'r') as f:
                        found_nmrduino = False
                        for line in f:
                            line = line.strip()
                            if '[NMRduino]' in line:
                                found_nmrduino = True
                            elif found_nmrduino and 'SampleRate' in line:
                                # format: SampleRate=6000
                                parts = line.split('=')
                                if len(parts) > 1:
                                    sampling_rate = float(parts[1])
                                break
                except Exception as e:
                    print(f"Warning: Could not read 0.ini: {e}")
                
                # If we found a custom one, break, otherwise keep looking or stick to default
                if sampling_rate != DEFAULT_SAMPLING_RATE:
                    break
        
        return sampling_rate

    def get_full_average(self):
        """
        Computes the average of all available scans.
        Optimized to not store all individual scans in memory if possible, 
        but load_all_scans does store them.
        Here we use a running sum for memory efficiency.
        """
        sum_buffer = None
        count = 0
        
        for fpath in self.scan_files:
            data = self._read_single_scan(fpath)
            if data is None:
                continue
                
            if sum_buffer is None:
                sum_buffer = np.zeros_like(data)
                
            if len(data) != len(sum_buffer):
                 min_len = min(len(data), len(sum_buffer))
                 sum_buffer = sum_buffer[:min_len] + data[:min_len]
                 if len(sum_buffer) > min_len:
                      sum_buffer = sum_buffer[:min_len]
            else:
                sum_buffer += data
                
            count += 1
            
        if count == 0:
            return 0, None
            
        return count, sum_buffer / count

    def _read_single_scan(self, file_path):
        """
        Reads a single .dat file.
        Logic adapted from nmrduino_util_fixed.py
        """
        try:
            with open(file_path, 'rb') as file:
                byte_data = bytearray(file.read())
            
            # Reverse byte order (Little Endian conversion for NMRDuino data)
            byte_data.reverse()
            
            # Unpack as 16-bit integers
            # Each int16 is 2 bytes
            num_points = len(byte_data) // 2
            int16_data = struct.unpack(f'<{num_points}h', byte_data)
            
            # Skip first 20 (header/garbage) and last 2 (footer)
            # This is standard behavior for the reference hardware
            if len(int16_data) > 22:
                np_data = np.array(int16_data[20:-2], dtype=np.float64) # Use float for processing
                # Correct time axis direction (reverse reversed buffer)
                np_data = np.flip(np_data)
            else:
                return None
                
            return np_data
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

    def generate_checkpoints(self, num_points=20):
        """
        Generates a list of scan counts (Checkpoints) uniformly distributed 
        on a square root scale.
        
        Args:
            num_points (int): Number of checkpoints desired.
            
        Returns:
            list[int]: Sorted list of N (scan counts).
        """
        if self.total_scans == 0:
            return []
            
        # Strategy: Uniform distribution in sqrt space
        # Start from sqrt(1) to sqrt(Total)
        sqrt_max = np.sqrt(self.total_scans)
        
        # If total scans is small, just return 1..Total
        if self.total_scans <= num_points:
            return list(range(1, self.total_scans + 1))
            
        sqrt_steps = np.linspace(1.0, sqrt_max, num_points)
        
        # Square back to get N, floor to int, unique to remove duplicates
        checkpoints = np.unique(np.floor(sqrt_steps**2).astype(int))
        
        # Filter out 0 if any
        checkpoints = checkpoints[checkpoints > 0]
        
        # Ensure the final total is included
        if len(checkpoints) == 0 or checkpoints[-1] != self.total_scans:
            if len(checkpoints) > 0 and checkpoints[-1] == self.total_scans:
                pass 
            else:
                # Append if not present, or replace last if very close? 
                # Appending is safer.
                checkpoints = np.append(checkpoints, self.total_scans)
                
        # Ensure sorted and unique again just in case
        return sorted(np.unique(checkpoints).tolist())

    def stream_process(self, checkpoints):
        """
        Generator that yields averaged data for each checkpoint.
        Uses 'Running Sum' to avoid reloading files.
        
        Yields:
            tuple: (n_scans, averaged_fid_data)
        """
        if not checkpoints:
            return

        sorted_checkpoints = sorted(checkpoints)
        checkpoint_set = set(sorted_checkpoints)
        max_needed = sorted_checkpoints[-1]
        
        sum_buffer = None
        current_count = 0
        
        for i, fpath in enumerate(self.scan_files):
            # i is 0-based index, scan count is i+1
            scan_idx = i + 1
            
            if scan_idx > max_needed:
                break
                
            data = self._read_single_scan(fpath)
            
            if data is None:
                continue
                
            if sum_buffer is None:
                sum_buffer = np.zeros_like(data)
                
            # Handle length mismatch (rare, but possible if settings changed mid-expt)
            if len(data) != len(sum_buffer):
                # Accessing len of min length
                min_len = min(len(data), len(sum_buffer))
                sum_buffer = sum_buffer[:min_len] + data[:min_len]
                # If sum buffer was longer, truncate
                if len(sum_buffer) > min_len:
                     sum_buffer = sum_buffer[:min_len]
            else:
                sum_buffer += data
                
            current_count += 1
            
            if current_count in checkpoint_set:
                # Compute average
                avg_data = sum_buffer / current_count
                yield current_count, avg_data

    def load_all_scans(self):
        """
        Load all scans into a 2D array (Scans x Points).
        Useful for Bootstrap mode or advanced analysis.
        
        Returns:
            np.ndarray: shape (N_scans, N_points)
        """
        all_data = []
        for fpath in self.scan_files:
            data = self._read_single_scan(fpath)
            if data is not None:
                all_data.append(data)
        
        if not all_data:
            return None
            
        # Find minimum length to stack safely
        min_len = min(len(d) for d in all_data)
        
        # Stack and truncate to min length
        stacked_data = np.vstack([d[:min_len] for d in all_data])
        return stacked_data
