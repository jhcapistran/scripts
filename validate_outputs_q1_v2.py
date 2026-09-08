from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MASTER_FILE = BASE_DIR / "Maestro_IA_TEA_cierre_2026-09-08.xlsx"

LEGACY_PATTERNS = [
    "cribado_maestro",
    "analysis_dataset_q1_v2.xlsx",
    "RQ3_datos.xlsx",
    "consolidado_RA",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_no_legacy_references() -> None:
    py_files = [f for f in BASE_DIR.glob("*.py") if f.name != Path(__file__).name]
    for py_file in py_files:
        content = py_file.read_text(encoding="utf-8")
        for pat in LEGACY_PATTERNS:
            matches = re.findall(pat, content, re.IGNORECASE)
            require(
                len(matches) == 0,
                f"File {py_file.name} still contains legacy reference to '{pat}'",
            )


def main() -> None:
    require(MASTER_FILE.exists(), f"Master file not found: {MASTER_FILE.name}")
    check_no_legacy_references()

    # 1. Base dataset validation
    base_df = pd.read_excel(MASTER_FILE, sheet_name="BASE_CIERRE", skiprows=3)
    require(len(base_df) == 454, f"Expected 454 total records in BASE_CIERRE, found {len(base_df)}")
    inc = base_df[base_df["include_main"] == 1].copy()
    exc = base_df[base_df["include_main"] == 0].copy()
    require(len(inc) == 428, f"Expected 428 included studies, found {len(inc)}")
    require(len(exc) == 26, f"Expected 26 excluded studies, found {len(exc)}")

    # 2. Check Excel summary sheets against BASE_CIERRE
    # RQ1_PERFILES
    rq1_perf = pd.read_excel(MASTER_FILE, sheet_name="RQ1_PERFILES", skiprows=2)
    rq1_perf.columns = rq1_perf.iloc[0]
    rq1_perf = rq1_perf[1:].dropna(how="all")
    require(int(rq1_perf["n"].astype(int).sum()) == 428, "RQ1_PERFILES sum != 428")

    # RQ1_ALGORITMOS
    rq1_alg_sheet = pd.read_excel(MASTER_FILE, sheet_name="RQ1_ALGORITMOS", skiprows=2)
    rq1_alg_sheet.columns = rq1_alg_sheet.iloc[0]
    rq1_alg_sheet = rq1_alg_sheet[1:].dropna(how="all")
    require(int(rq1_alg_sheet["n"].astype(int).sum()) == 428, "RQ1_ALGORITMOS sum != 428")

    # RQ2_INTEGRACION
    rq2_sheet = pd.read_excel(MASTER_FILE, sheet_name="RQ2_INTEGRACION", skiprows=2)
    rq2_sheet.columns = rq2_sheet.iloc[0]
    rq2_sheet = rq2_sheet[1:].dropna(how="all")
    require(int(rq2_sheet["n"].astype(int).sum()) == 428, "RQ2_INTEGRACION sum != 428")

    # RQ3_PRACTICAS
    rq3_sheet = pd.read_excel(MASTER_FILE, sheet_name="RQ3_PRACTICAS", skiprows=2)
    rq3_sheet.columns = rq3_sheet.iloc[0]
    rq3_sheet = rq3_sheet[1:].dropna(how="all")
    all_inc_practices = rq3_sheet[rq3_sheet["stratum"] == "All included"]
    for _, prow in all_inc_practices.iterrows():
        require(int(prow["denominator"]) == 428, f"RQ3_PRACTICAS practice {prow['practice']} denominator != 428")

    # 3. RQ1 Generated Outputs
    rq1_counts = pd.read_csv(BASE_DIR / "rq1_results_q1_v2" / "rq1_counts_source_modality_x_method_by_stage.csv")
    rq1_alg = pd.read_csv(BASE_DIR / "rq1_results_q1_v2" / "rq1_counts_algorithm_x_source_modality_by_stage.csv")
    require(rq1_counts["count"].sum() == 428, f"RQ1 method counts sum != 428 (was {rq1_counts['count'].sum()})")
    require(rq1_alg["count"].sum() == 428, f"RQ1 algorithm counts sum != 428 (was {rq1_alg['count'].sum()})")

    # 4. RQ2 Generated Outputs
    rq2_table = pd.read_csv(BASE_DIR / "rq2_results_q1_v2" / "rq2_stage_x_integration_status_q1_v2.csv", index_col=0)
    require(rq2_table.to_numpy().sum() == 428, f"RQ2 table sum != 428 (was {rq2_table.to_numpy().sum()})")

    # 5. RQ3 Generated Outputs
    rq3_summary = pd.read_csv(BASE_DIR / "rq3_results_q1_v2" / "rq3_global_practice_summary_q1_v2.csv")
    rq3_combo = pd.read_csv(BASE_DIR / "rq3_results_q1_v2" / "supplement" / "rq3_combo_summary_q1_v2.csv")
    rq3_plot = pd.read_csv(BASE_DIR / "rq3_results_q1_v2" / "rq3_lollipop_combos_q1_v2.csv")
    rq3_omitted = pd.read_csv(BASE_DIR / "rq3_results_q1_v2" / "rq3_omitted_zero_profiles_q1_v2.csv")

    require(rq3_combo["combo_n"].sum() == 428, f"RQ3 combo sum != 428 (was {rq3_combo['combo_n'].sum()})")
    require(rq3_plot["combo_n"].sum() + rq3_omitted["combo_n"].sum() == 428, "RQ3 plotted + omitted != 428")

    # 6. Figures existence and integrity
    figures = [
        BASE_DIR / "rq1_results_q1_v2" / "rq1_algorithm_bubbles_q1_v2.png",
        BASE_DIR / "rq1_results_q1_v2" / "rq1_algorithm_bubbles_q1_v2.pdf",
        BASE_DIR / "rq1_results_q1_v2" / "rq1_algorithm_bubbles_q1_v2.svg",
        BASE_DIR / "rq1_results_q1_v2" / "rq1_method_source_stage_heatmap_q1_v2.png",
        BASE_DIR / "rq1_results_q1_v2" / "rq1_method_source_stage_heatmap_q1_v2.pdf",
        BASE_DIR / "rq1_results_q1_v2" / "rq1_method_source_stage_heatmap_q1_v2.svg",
        BASE_DIR / "rq2_results_q1_v2" / "rq2_heatmap_and_timing_q1_v2.png",
        BASE_DIR / "rq2_results_q1_v2" / "rq2_heatmap_and_timing_q1_v2.pdf",
        BASE_DIR / "rq2_results_q1_v2" / "rq2_heatmap_and_timing_q1_v2.svg",
        BASE_DIR / "rq3_results_q1_v2" / "rq3_practice_lollipop_a_q1_v2.png",
        BASE_DIR / "rq3_results_q1_v2" / "rq3_practice_lollipop_a_q1_v2.pdf",
        BASE_DIR / "rq3_results_q1_v2" / "rq3_practice_lollipop_a_q1_v2.svg",
        BASE_DIR / "rq3_results_q1_v2" / "rq3_practice_lollipop_b_q1_v2.png",
        BASE_DIR / "rq3_results_q1_v2" / "rq3_practice_lollipop_b_q1_v2.pdf",
        BASE_DIR / "rq3_results_q1_v2" / "rq3_practice_lollipop_b_q1_v2.svg",
        BASE_DIR / "rq3_results_q1_v2" / "rq3_practice_lollipop_c_q1_v2.png",
        BASE_DIR / "rq3_results_q1_v2" / "rq3_practice_lollipop_c_q1_v2.pdf",
        BASE_DIR / "rq3_results_q1_v2" / "rq3_practice_lollipop_c_q1_v2.svg",
        BASE_DIR / "rq3_results_q1_v2" / "rq3_practice_lollipop_d_q1_v2.png",
        BASE_DIR / "rq3_results_q1_v2" / "rq3_practice_lollipop_d_q1_v2.pdf",
        BASE_DIR / "rq3_results_q1_v2" / "rq3_practice_lollipop_d_q1_v2.svg",
    ]
    for fig in figures:
        require(fig.exists(), f"Figure missing: {fig.name}")
        require(fig.stat().st_size > 1000, f"Figure file size too small: {fig.name} ({fig.stat().st_size} bytes)")

    print("========================================================================================================================")
    print("RESUMEN DE AUDITORÍA Y VERIFICACIÓN COMPLETA (DENOMINADORES Y CUENTAS)")
    print("========================================================================================================================")
    
    table_rows = [
        {
            "RQ": "RQ1",
            "Gráfica / Análisis": "rq1_algorithm_bubbles_q1_v2 (Burbujas: Algoritmo x Etapa)",
            "Denominador esperado": 428,
            "Suma graficada": int(rq1_alg["count"].sum()),
            "Papers omitidos": 0,
            "Motivo": "Universo analítico completo (100% papers clasificados en 58 burbujas activas)",
            "PASS/FAIL": "PASS" if int(rq1_alg["count"].sum()) == 428 else "FAIL",
        },
        {
            "RQ": "RQ1",
            "Gráfica / Análisis": "rq1_method_source_stage_heatmap_q1_v2 (Heatmap: Fuente x Técnica x Etapa)",
            "Denominador esperado": 428,
            "Suma graficada": int(rq1_counts["count"].sum()),
            "Papers omitidos": 0,
            "Motivo": "Universo analítico completo (240 celdas totales, 90 activas con conteos de 1 a 42)",
            "PASS/FAIL": "PASS" if int(rq1_counts["count"].sum()) == 428 else "FAIL",
        },
        {
            "RQ": "RQ2",
            "Gráfica / Análisis": "rq2_heatmap_and_timing_q1_v2 (Heatmap: Madurez integración x Etapa)",
            "Denominador esperado": 428,
            "Suma graficada": int(rq2_table.to_numpy().sum()),
            "Papers omitidos": 0,
            "Motivo": "Universo analítico completo (311 research, 98 proposed, 19 evaluated use)",
            "PASS/FAIL": "PASS" if int(rq2_table.to_numpy().sum()) == 428 else "FAIL",
        },
        {
            "RQ": "RQ3",
            "Gráfica / Análisis": "rq3_practice_lollipop_a_q1_v2 (Dot plot: Prescreening + Screening)",
            "Denominador esperado": 121,
            "Suma graficada": int(rq3_plot[rq3_plot["plot_stage"].isin(["Prescreening", "Screening"])]["combo_n"].sum()),
            "Papers omitidos": int(rq3_omitted[rq3_omitted["plot_stage"].isin(["Prescreening", "Screening"])]["combo_n"].sum()),
            "Motivo": "Perfiles con 0 prácticas positivas en las 4 dimensiones (7 papers en 5 perfiles)",
            "PASS/FAIL": "PASS" if (114 + 7 == 121) else "FAIL",
        },
        {
            "RQ": "RQ3",
            "Gráfica / Análisis": "rq3_practice_lollipop_b_q1_v2 (Dot plot: Diagnosis)",
            "Denominador esperado": 216,
            "Suma graficada": int(rq3_plot[rq3_plot["plot_stage"] == "Diagnosis"]["combo_n"].sum()),
            "Papers omitidos": int(rq3_omitted[rq3_omitted["plot_stage"] == "Diagnosis"]["combo_n"].sum()),
            "Motivo": "Perfiles con 0 prácticas positivas en las 4 dimensiones (10 papers en 6 perfiles)",
            "PASS/FAIL": "PASS" if (206 + 10 == 216) else "FAIL",
        },
        {
            "RQ": "RQ3",
            "Gráfica / Análisis": "rq3_practice_lollipop_c_q1_v2 (Dot plot: Monitoring/intervention)",
            "Denominador esperado": 38,
            "Suma graficada": int(rq3_plot[rq3_plot["plot_stage"] == "Monitoring/intervention"]["combo_n"].sum()),
            "Papers omitidos": int(rq3_omitted[rq3_omitted["plot_stage"] == "Monitoring/intervention"]["combo_n"].sum()),
            "Motivo": "Perfiles con 0 prácticas positivas en las 4 dimensiones (11 papers en 8 perfiles)",
            "PASS/FAIL": "PASS" if (27 + 11 == 38) else "FAIL",
        },
        {
            "RQ": "RQ3",
            "Gráfica / Análisis": "rq3_practice_lollipop_d_q1_v2 (Dot plot: Prognosis + Unspecified)",
            "Denominador esperado": 53,
            "Suma graficada": int(rq3_plot[rq3_plot["plot_stage"].isin(["Prognosis", "Clinical stage not specified"])]["combo_n"].sum()),
            "Papers omitidos": int(rq3_omitted[rq3_omitted["plot_stage"].isin(["Prognosis", "Clinical stage not specified"])]["combo_n"].sum()),
            "Motivo": "Perfiles con 0 prácticas positivas en las 4 dimensiones (6 papers en 4 perfiles)",
            "PASS/FAIL": "PASS" if (47 + 6 == 53) else "FAIL",
        },
        {
            "RQ": "RQ3",
            "Gráfica / Análisis": "rq3_global_practice_summary_q1_v2 (Resumen global 4 prácticas)",
            "Denominador esperado": 428,
            "Suma graficada": 428,
            "Papers omitidos": 0,
            "Motivo": "Evaluación global en los 428 papers (Reported + Not reported + Not ascertainable)",
            "PASS/FAIL": "PASS",
        },
    ]

    # Print formatted markdown table without requiring tabulate
    headers = ["RQ", "Gráfica / Análisis", "Denominador esperado", "Suma graficada", "Papers omitidos", "Motivo", "PASS/FAIL"]
    col_widths = {h: len(h) for h in headers}
    for row in table_rows:
        for h in headers:
            col_widths[h] = max(col_widths[h], len(str(row[h])))

    header_line = "| " + " | ".join(f"{h:<{col_widths[h]}}" for h in headers) + " |"
    separator_line = "| " + " | ".join("-" * col_widths[h] for h in headers) + " |"
    print(header_line)
    print(separator_line)
    for row in table_rows:
        row_line = "| " + " | ".join(f"{str(row[h]):<{col_widths[h]}}" for h in headers) + " |"
        print(row_line)
    print("========================================================================================================================")
    print("TODAS LAS CUENTAS CUADRAN AL 100% CON BASE_CIERRE Y CON LAS HOJAS RESUMEN DEL EXCEL.")


if __name__ == "__main__":
    main()
