#!/usr/bin/env python3
"""Start the toio operator with web UI."""

import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from toio_bc.operator import main

if __name__ == "__main__":
    main()
