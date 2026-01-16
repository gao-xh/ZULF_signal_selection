import sys
import os

# Ensure the project root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ui_main import main

if __name__ == "__main__":
    main()
