#!/usr/bin/env python3
"""Train a policy model."""

import sys
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from toio_bc.train import main

if __name__ == "__main__":
    main()
