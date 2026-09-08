from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MASTER_FILE = BASE_DIR / "cribado_maestro_276_actualizacion_FINAL_CORREGIDO_2026-09-07.xlsx"
DATASET = BASE_DIR / "analysis_dataset_q1_v2.xlsx"

BINARY_COLS = [
    "ds_neuroimaging",
    "ds_physiological",
    "ds_behavioral_video",
    "ds_voice_audio",
    "ds_structured_records",
    "ds_other_biological",
    "ds_other",
    "stage_prescreening",
    "stage_screening",
    "stage_diagnosis",
    "stage_prognosis",
    "stage_monitoring_intervention",
    "q3_external_validation",
    "q3_multisite_dataset",
    "q3_cross_site_robustness",
    "q3_prospective_evaluation",
    "q3_xai_strict",
    "q3_xai_partial",
    "q3_multisource_data",
]

REQUIRED_RQ_COLS = [
    "modalidad",
    "tipo_IA",
    "AI_algorithm_main",
    "AI_task_type",
    "stage_primary",
    "rq2_integration_status",
    "rq2_role",
    "q3_external_validation",
    "q3_multisite_dataset",
    "q3_cross_site_robustness",
    "q3_prospective_evaluation",
    "q3_xai_strict",
    "q3_xai_partial",
    "q3_multisource_data",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read(sheet: str) -> pd.DataFrame:
    return pd.read_excel(DATASET, sheet_name=sheet)


def check_binary_columns(df: pd.DataFrame, sheet: str) -> None:
    for col in BINARY_COLS:
        if col not in df.columns:
            continue
        values = set(df[col].dropna().unique().tolist())
        require(values <= {0, 1}, f"{sheet}.{col} is not binary 0/1: {sorted(values)[:10]}")


def main() -> None:
    require(MASTER_FILE.exists(), "corrected master workbook does not exist")
    require(DATASET.exists(), "analysis_dataset_q1_v2.xlsx does not exist")

    master = pd.ExcelFile(MASTER_FILE)
    for sheet in ["RQ1_RQ2_base_276", "RQ3_base_276", "Texto_completo_191", "Extraccion_RQ_nuevos"]:
        require(sheet in master.sheet_names, f"Master is missing {sheet}")

    master_rq12 = pd.read_excel(MASTER_FILE, sheet_name="RQ1_RQ2_base_276")
    master_rq3 = pd.read_excel(MASTER_FILE, sheet_name="RQ3_base_276")
    extraction = pd.read_excel(MASTER_FILE, sheet_name="Extraccion_RQ_nuevos")
    text_complete = pd.read_excel(MASTER_FILE, sheet_name="Texto_completo_191")

    historical_n = len(master_rq12)
    require(historical_n == len(master_rq3), "Historical RQ1/RQ2 and RQ3 sheet lengths differ")
    require(master_rq12["study_id"].nunique() == historical_n, "Historical RQ1/RQ2 study_id values are not unique")
    require(master_rq3["study_id"].nunique() == historical_n, "Historical RQ3 study_id values are not unique")

    new_included_n = int(extraction["full_text_decision"].isin(["Include", "Include with integrity flag"]).sum())
    excluded_n = int(text_complete["full_text_decision"].eq("Exclude").sum())
    pending_n = int(text_complete["full_text_decision"].astype(str).str.contains("Pending", case=False, na=False).sum())

    rq12 = read("rq1_rq2_graph_ready")
    rq3 = read("rq3_graph_ready")
    included = read("included_studies")
    prisma = read("PRISMA_scope")

    final_n = historical_n + new_included_n
    require(final_n == 454, f"Expected 454 unique studies = historical + new, got {final_n}")
    require(historical_n == 276, f"Expected 276 historical studies, got {historical_n}")
    require(new_included_n == 178, f"Expected 178 new included studies, got {new_included_n}")
    require(excluded_n == 14, f"Expected 14 full-text exclusions, got {excluded_n}")
    require(pending_n == 0, f"Expected 0 pending full-text adjudications, got {pending_n}")

    for df, sheet in [(rq12, "rq1_rq2_graph_ready"), (rq3, "rq3_graph_ready")]:
        require(len(df) == final_n, f"{sheet} does not contain {final_n} rows")
        require(df["study_id"].nunique() == final_n, f"{sheet} does not contain {final_n} unique study_id values")
        require(set(REQUIRED_RQ_COLS) <= set(df.columns), f"{sheet} is missing required normalized columns")
        check_binary_columns(df, sheet)
    require(len(included) == final_n, f"included_studies does not contain {final_n} rows")
    require(included["study_id"].nunique() == final_n, f"included_studies does not contain {final_n} unique study_id values")

    require(rq12["study_id"].tolist() == rq3["study_id"].tolist(), "RQ1/RQ2 and RQ3 study order differs")
    require(int(included["cohort"].eq("historical_276").sum()) == historical_n, "included_studies historical count mismatch")
    require(int(included["cohort"].eq("update_2026_included").sum()) == new_included_n, "included_studies new count mismatch")
    require(int(prisma.loc[prisma["stage"].eq("Final analytical corpus"), "n"].iloc[0]) == final_n, "PRISMA final corpus count mismatch")
    require(int(prisma.loc[prisma["stage"].eq("Full-text exclusions"), "n"].iloc[0]) == excluded_n, "PRISMA exclusion count mismatch")
    require(int(prisma.loc[prisma["stage"].eq("Pending full-text adjudication"), "n"].iloc[0]) == pending_n, "PRISMA pending count mismatch")

    rq2_denoms = pd.read_csv(BASE_DIR / "rq2_results_q1_v2" / "rq2_denominators_q1_v2.csv")
    require("title_abstract_signal_state" not in set(rq2_denoms["subset"]), "RQ2 still validates legacy title/abstract candidate signals")
    require(set(rq2_denoms["subset"]) >= {"all_rows", "integration_status"}, "RQ2 denominators do not use integration_status")
    require(not (BASE_DIR / "rq2_results_q1_v2" / "rq2_stage_x_preliminary_signal_q1_v2.csv").exists(), "Stale RQ2 preliminary-signal CSV still exists")
    require((BASE_DIR / "rq2_results_q1_v2" / "rq2_stage_x_integration_status_q1_v2.csv").exists(), "RQ2 integration-status CSV was not written")
    caption = (BASE_DIR / "rq2_results_q1_v2" / "rq2_heatmap_and_timing_q1_v2_caption.txt").read_text(encoding="utf-8")
    require("q2_candidate_abstract and q2_candidate_terms are not treated as final" in caption, "RQ2 caption does not reject legacy q2_candidate_* final use")

    rq3_summary = pd.read_csv(BASE_DIR / "rq3_results_q1_v2" / "rq3_global_practice_summary_q1_v2.csv")
    require(set(rq3_summary["practice_signal"]) == {"q3_external_validation", "q3_multisource_data", "q3_xai_any", "q3_site_any"}, "RQ3 summary is not based on new q3_* fields")

    figure_paths = [
        BASE_DIR / "rq1_results_q1_v2" / "rq1_algorithm_bubbles_q1_v2.png",
        BASE_DIR / "rq1_results_q1_v2" / "rq1_method_source_stage_heatmap_q1_v2.png",
        BASE_DIR / "rq2_results_q1_v2" / "rq2_heatmap_and_timing_q1_v2.png",
        BASE_DIR / "rq3_results_q1_v2" / "rq3_practice_lollipop_a_q1_v2.png",
        BASE_DIR / "rq3_results_q1_v2" / "rq3_practice_lollipop_b_q1_v2.png",
        BASE_DIR / "rq3_results_q1_v2" / "rq3_practice_lollipop_c_q1_v2.png",
        BASE_DIR / "rq3_results_q1_v2" / "rq3_practice_lollipop_d_q1_v2.png",
    ]
    for path in figure_paths:
        require(path.exists() and path.stat().st_size > 10_000, f"Missing or tiny figure: {path.name}")

    print(f"PASS: {final_n} unique = {historical_n} + {new_included_n}; full-text exclusions={excluded_n}; pending={pending_n}.")


if __name__ == "__main__":
    main()
