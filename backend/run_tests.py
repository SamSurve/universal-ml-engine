import sys
import pytest
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    test_dir = str(project_root / "backend" / "tests")
    print(f"Running pytest programmatically on: {test_dir}")
    exit_code = pytest.main(["-v", test_dir])
    print(f"\nPytest Exit Code: {exit_code}")
    sys.exit(exit_code)
