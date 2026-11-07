"""Put the service root on sys.path for direct pytest invocation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
