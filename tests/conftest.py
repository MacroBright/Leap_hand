"""pytest configuration for leap_hand test suite."""
import sys
from pathlib import Path

# Ensure src/ and legacy python/ are resolvable
ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
PYTHON_DIR = ROOT / "python"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(1, str(PYTHON_DIR))
