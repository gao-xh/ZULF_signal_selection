# Data Loading Interface Architecture

This document outlines the architecture used for the data loading interface in the ZULF Signal Selection application. It is designed to be robust, user-friendly, and decoupled, separating UI interaction from low-level file handling.

## 1. Architecture Overview

The system is divided into two distinct layers:
1.  **UI Layer (`ui_main.py`)**: Manages user interaction, allowing selection of "Experiment Folders" rather than individual files.
2.  **Logic Layer (`loader.py`)**: Handles file discovery, sorting, aggregation, and binary parsing.

### Key Benefits
*   **Abstraction**: Users operate on *Experiments* (Folders), not raw bytes.
*   **Multi-Dataset Support**: Seamlessly combines data from multiple partial experiments (e.g., `run_01`, `run_02`) into a single dataset.
*   **Resilience**: Gracefully handles missing files or interrupted experiment runs.
*   **Decoupling**: File format changes (e.g., `.dat` to `.csv`) only require changes in the Loader, not the UI.

---

## 2. Frontend Interface (UI Layer)

**File:** `src/ui_main.py`

The UI provides a simple list-management interface. It maintains a list of directory paths which are passed to the backend.

### Visual Components
*   **`QListWidget`**: Displays the queue of selected folders.
*   **`QFileDialog`**: Standard system dialog for directory selection.
*   **Action Buttons**: "Add Folder" and "Clear".

### Implementation Pattern

```python
# --- Layout Setup ---
folder_group = QGroupBox("Data Selection")
layout = QVBoxLayout()

# List Widget to show selected folders
self.folder_list = QListWidget()
self.folder_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
layout.addWidget(self.folder_list)

# Buttons
btn_layout = QHBoxLayout()
self.btn_add_folder = QPushButton("Add Folder(s)")
self.btn_add_folder.clicked.connect(self.add_folders) # Connects to handler
# ... (Clear button setup) ...

# --- Event Handler Logic ---
def add_folders(self):
    """
    Opens system dialog to select a directory.
    Adds it to the internal list and UI list if not already present.
    """
    folder = QFileDialog.getExistingDirectory(self, "Select Experiment Folder")
    if folder:
        path = str(Path(folder).resolve())
        if path not in self.folder_paths:
            self.folder_paths.append(path)
            self.folder_list.addItem(path)
```

---

## 3. Backend Logic (Loader Layer)

**File:** `src/loader.py`

The `ProgressiveLoader` class acts as the bridge. It accepts a list of folder paths and abstracts them into a continuous stream of signal data.

### Core Responsibilities
1.  **Aggregation**: Iterates through all provided folders.
2.  **Natural Sorting**: Ensures `10.dat` comes after `2.dat` (Numerical Sort vs Alphabetical).
3.  **Metadata Extraction**: Automatically hunts for `0.ini` to find sampling rates.
4.  **Binary Parsing**: encapsulating endianness and array flipping.

### Implementation Pattern

```python
class ProgressiveLoader:
    def __init__(self, folder_paths):
        # normalize input to list of Path objects
        self.folder_paths = [Path(p).resolve() for p in folder_paths]
        self.scan_files = []

        # 1. Aggregation Loop
        for folder in self.folder_paths:
            if not folder.exists(): continue
                
            # 2. File Discovery
            folder_files = list(folder.glob("*.dat"))
            valid_files = [f for f in folder_files if f.stem.isdigit()]
            
            # 3. Natural Sorting (Key Step)
            # Sorts by integer value of filename (1, 2, ... 10)
            valid_files.sort(key=lambda f: int(f.stem))
            
            self.scan_files.extend(valid_files)
        
        # 4. Metadata Lookup
        self.sampling_rate = self._read_sampling_rate()

    def _read_single_scan(self, file_path):
        """
        Encapsulated binary reading logic.
        UI never sees 'struct.unpack' or 'bytearray'.
        """
        try:
            with open(file_path, 'rb') as f:
                # Read raw bytes
                byte_data = bytearray(f.read())
            
            # Specialized parsing logic (e.g., reverse bytes, unpack int16)
            # ...
            return np_data
        except Exception as e:
            return None
```
