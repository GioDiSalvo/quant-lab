import sys
from pathlib import Path

# make `import quantlab` work when running `pytest` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
