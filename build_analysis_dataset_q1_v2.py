from __future__ import annotations

"""
build_analysis_dataset_q1_v2.py

Nota de cierre metodológico (2026-09-08):
Los scripts de análisis (rq1_script.py, rq2_script.py, rq3_script.py) ahora leen directamente
la hoja BASE_CIERRE del archivo congelado Maestro_IA_TEA_cierre_2026-09-08.xlsx como única fuente de verdad.
No se requiere generar datasets intermedios ni aplicar heurísticas de imputación.
"""

from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MASTER_FILE = BASE_DIR / "Maestro_IA_TEA_cierre_2026-09-08.xlsx"


def main() -> None:
    if not MASTER_FILE.exists():
        raise FileNotFoundError(f"Master file not found: {MASTER_FILE}")
    print(f"Archivo maestro vigente: {MASTER_FILE.name}")
    df = pd.read_excel(MASTER_FILE, sheet_name="BASE_CIERRE", skiprows=3)
    included = df[df["include_main"] == 1]
    excluded = df[df["include_main"] == 0]
    print(f"Total registros conservados: {len(df)}")
    print(f"Incluidos en síntesis principal: {len(included)}")
    print(f"Exclusiones de auditoría: {len(excluded)}")
    print("Los scripts RQ1-RQ3 leen directamente BASE_CIERRE de este maestro final.")


if __name__ == "__main__":
    main()
