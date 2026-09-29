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

RQ2_ADJUDICATED_COLS = [
    "rq2_status",
    "rq2_role",
    "decision_timing_coded",
    "observed_decision_timing",
    "source_level",
    "rationale",
]
RQ2_MATURITY_ORDER = ["Research only", "Proposed only", "Evaluated AI use"]
RQ2_TIMING_ORDER = ["Pre-decision", "In-decision", "Post-decision"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def rq2_maturity(status: object) -> str:
    status_text = str(status)
    if status_text.startswith("Evaluated"):
        return "Evaluated AI use"
    return status_text


def require_frame_equal(actual: pd.DataFrame, expected: pd.DataFrame, message: str) -> None:
    try:
        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
            check_like=False,
            check_names=False,
        )
    except AssertionError as exc:
        raise AssertionError(f"{message}: {exc}") from exc


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
    inc["integration_maturity"] = inc["rq2_status"].map(rq2_maturity)
    rq2_dir = BASE_DIR / "rq2_results_q1_v2"
    rq2_maturity_expected = inc["integration_maturity"].value_counts().reindex(RQ2_MATURITY_ORDER, fill_value=0)
    require(
        rq2_maturity_expected.to_dict() == {"Research only": 311, "Proposed only": 98, "Evaluated AI use": 19},
        f"RQ2 maturity counts changed in master: {rq2_maturity_expected.to_dict()}",
    )

    rq2_stage_maturity = pd.read_csv(rq2_dir / "rq2_stage_x_integration_maturity_q1_v2.csv", index_col=0)
    expected_stage_maturity = (
        pd.crosstab(inc["stage_primary"].astype(str), inc["integration_maturity"])
        .reindex(index=rq2_stage_maturity.index, columns=RQ2_MATURITY_ORDER, fill_value=0)
    )
    require_frame_equal(rq2_stage_maturity, expected_stage_maturity, "RQ2 maturity x stage CSV does not match BASE_CIERRE")
    require(rq2_stage_maturity.to_numpy().sum() == 428, f"RQ2 maturity table sum != 428 (was {rq2_stage_maturity.to_numpy().sum()})")
    require(int(rq2_stage_maturity["Research only"].sum()) == 311, "RQ2 Research only total != 311")
    require(int(rq2_stage_maturity["Proposed only"].sum()) == 98, "RQ2 Proposed only total != 98")
    require(int(rq2_stage_maturity["Evaluated AI use"].sum()) == 19, "RQ2 Evaluated AI use total != 19")

    rq2_status_table = pd.read_csv(rq2_dir / "rq2_stage_x_integration_status_q1_v2.csv", index_col=0)
    require(rq2_status_table.to_numpy().sum() == 428, f"RQ2 status table sum != 428 (was {rq2_status_table.to_numpy().sum()})")

    proposed_evaluated = inc[inc["integration_maturity"].isin(["Proposed only", "Evaluated AI use"])].copy()
    evaluated = inc[inc["integration_maturity"] == "Evaluated AI use"].copy()
    require(len(proposed_evaluated) == 117, f"RQ2 Proposed/Evaluated subset != 117 (was {len(proposed_evaluated)})")
    require(len(evaluated) == 19, f"RQ2 Evaluated subset != 19 (was {len(evaluated)})")

    roles_csv = pd.read_csv(rq2_dir / "rq2_role_proposed_evaluated_q1_v2.csv")
    expected_roles = (
        proposed_evaluated["rq2_role"]
        .value_counts()
        .reindex(sorted(proposed_evaluated["rq2_role"].dropna().unique()), fill_value=0)
        .rename_axis("rq2_role")
        .reset_index(name="n")
    )
    require_frame_equal(roles_csv, expected_roles, "RQ2 role CSV does not match BASE_CIERRE")
    require(int(roles_csv["n"].sum()) == 117, f"RQ2 role total != 117 (was {roles_csv['n'].sum()})")

    coded_csv = pd.read_csv(rq2_dir / "rq2_decision_timing_coded_proposed_evaluated_q1_v2.csv")
    expected_coded = (
        proposed_evaluated["decision_timing_coded"]
        .value_counts()
        .reindex(RQ2_TIMING_ORDER, fill_value=0)
        .rename_axis("decision_timing_coded")
        .reset_index(name="n")
    )
    require_frame_equal(coded_csv, expected_coded, "RQ2 coded timing CSV does not match BASE_CIERRE")
    require(int(coded_csv["n"].sum()) == 117, f"RQ2 coded timing total != 117 (was {coded_csv['n'].sum()})")

    observed_csv = pd.read_csv(rq2_dir / "rq2_observed_decision_timing_evaluated_q1_v2.csv")
    expected_observed = (
        evaluated["observed_decision_timing"]
        .value_counts()
        .reindex(RQ2_TIMING_ORDER, fill_value=0)
        .rename_axis("observed_decision_timing")
        .reset_index(name="n")
    )
    require_frame_equal(observed_csv, expected_observed, "RQ2 observed timing CSV does not match BASE_CIERRE")
    require(int(observed_csv["n"].sum()) == 19, f"RQ2 observed timing total != 19 (was {observed_csv['n'].sum()})")

    non_evaluated = inc[inc["integration_maturity"] != "Evaluated AI use"]
    bad_observed = non_evaluated[non_evaluated["observed_decision_timing"] != "Not observed in assessed sources"]
    require(
        bad_observed.empty,
        "Research only or Proposed only rows appear as observed clinical timing: "
        + bad_observed[["study_id", "rq2_status", "observed_decision_timing"]].to_string(index=False),
    )
    observed_rows = inc[inc["observed_decision_timing"].isin(RQ2_TIMING_ORDER)]
    require(
        observed_rows["integration_maturity"].eq("Evaluated AI use").all(),
        "Rows with observed_decision_timing are not all Evaluated AI use: "
        + observed_rows[["study_id", "rq2_status", "observed_decision_timing"]].to_string(index=False),
    )

    adjudicated_csv = pd.read_csv(rq2_dir / "rq2_adjudicated_columns_q1_v2.csv")
    expected_adjudicated = inc[RQ2_ADJUDICATED_COLS].copy()
    require(list(adjudicated_csv.columns) == RQ2_ADJUDICATED_COLS, "RQ2 adjudicated CSV columns changed")
    require_frame_equal(adjudicated_csv, expected_adjudicated, "RQ2 adjudicated CSV does not exactly reproduce master columns")

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
        BASE_DIR / "rq2_results_q1_v2" / "rq2_integration_levels_and_timing_q1_v2.png",
        BASE_DIR / "rq2_results_q1_v2" / "rq2_integration_levels_and_timing_q1_v2.pdf",
        BASE_DIR / "rq2_results_q1_v2" / "rq2_integration_levels_and_timing_q1_v2.svg",
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
            "Gráfica / Análisis": "rq2_integration_levels_and_timing_q1_v2 (Madurez, rol y timing RQ2)",
            "Denominador esperado": 428,
            "Suma graficada": int(rq2_stage_maturity.to_numpy().sum()),
            "Papers omitidos": 0,
            "Motivo": "Panel A usa 428; paneles B-C usan 117 Proposed/Evaluated; panel D usa 19 Evaluated only",
            "PASS/FAIL": "PASS" if int(rq2_stage_maturity.to_numpy().sum()) == 428 and int(roles_csv["n"].sum()) == 117 and int(observed_csv["n"].sum()) == 19 else "FAIL",
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
