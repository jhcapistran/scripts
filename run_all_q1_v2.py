from __future__ import annotations

import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def run_script(name: str) -> None:
    subprocess.run([sys.executable, str(BASE_DIR / name)], check=True)


def main() -> None:
    for script_name in ("build_analysis_dataset_q1_v2.py", "rq1_script.py", "rq2_script.py", "rq3_script.py"):
        run_script(script_name)


if __name__ == "__main__":
    main()
