"""Run isolated CPU checks; mocks must not leak between test processes."""
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
TESTS = ("random_reference", "random_decode", "random_runner", "request_rng",
         "s0_resources", "s1_state", "ngram", "qk_norm_dispatch", "random_position_cases")
for name in TESTS:
    subprocess.run([sys.executable, str(ROOT / "tests" / f"test_{name}.py")],
                   cwd=ROOT, check=True)
