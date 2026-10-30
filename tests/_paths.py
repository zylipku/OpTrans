"""Allow validation scripts to run directly without installing the package."""
from pathlib import Path
import sys

RELEASE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = RELEASE_ROOT/'experiment1/configs'
DEFAULT_OUTPUTS = RELEASE_ROOT/'experiment1/outputs'
if str(RELEASE_ROOT) not in sys.path:
    sys.path.insert(0, str(RELEASE_ROOT))
