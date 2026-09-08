from __future__ import annotations

import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def run_script(name: str) -> None:
    print(f"[RUNNING] {name}...")
    subprocess.run([sys.executable, str(BASE_DIR / name)], check=True)
    print(f"[COMPLETED] {name}\n")


def main() -> None:
    print("Regenerating RQ1, RQ2, and RQ3 from Maestro_IA_TEA_cierre_2026-09-08.xlsx...\n")
    for script_name in ("rq1_script.py", "rq2_script.py", "rq3_script.py"):
        run_script(script_name)
    print("All RQ scripts executed successfully.")


if __name__ == "__main__":
    main()
