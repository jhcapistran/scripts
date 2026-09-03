from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
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
    "q2_candidate_abstract",
    "q2_candidate_terms",
    "q3_external_validation_signal",
    "q3_explainability_signal",
    "q3_multisite_signal",
    "q3_multisource_strategy_signal",
    "needs_full_text_check",
    "q3_completed_manual_19",
    "q3_cross_site_robustness_signal",
    "q3_multisite_dataset_signal",
    "q3_internal_validation_signal",
    "q3_prospective_evaluation_signal",
    "q3_xai_strict_signal",
    "q3_xai_partial_signal",
    "q3_multisource_data_signal",
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
    require(DATASET.exists(), "analysis_dataset_q1_v2.xlsx does not exist")

    rq12 = read("rq1_rq2_graph_ready")
    rq3 = read("rq3_graph_ready")
    candidates = read("new_candidates_passed_192")
    prisma = read("PRISMA_scope_468")

    require(len(rq12) == 276 and rq12["study_id"].nunique() == 276, "RQ1/RQ2 corpus is not 276 unique studies")
    require(len(rq3) == 276 and rq3["study_id"].nunique() == 276, "RQ3 corpus is not 276 unique studies")
    require(rq12["study_id"].tolist() == rq3["study_id"].tolist(), "RQ1/RQ2 and RQ3 study order differs")

    require(len(candidates) == 192, "Expected 192 provisional candidates")
    row188 = candidates[candidates["bib_index"].eq(188)]
    require(len(row188) == 1, "bib_index 188 is not present exactly once among candidates")
    require(row188["full_text_decision"].iloc[0] == "Pending adjudication", "bib_index 188 has an inferred full-text decision")

    for df, sheet in [(rq12, "rq1_rq2_graph_ready"), (rq3, "rq3_graph_ready")]:
        check_binary_columns(df, sheet)
        deep = df[df["title"].astype(str).str.contains("DeepASDPred", case=False, na=False)]
        require(len(deep) == 1, f"{sheet} does not contain exactly one DeepASDPred row")
        deep = deep.iloc[0]
        require(deep["modalidad"] == "Biological/omics", f"{sheet} DeepASDPred modality not corrected")
        require(deep["AI_task_type"] == "risk-RNA identification", f"{sheet} DeepASDPred task not corrected")
        require(deep["stage_primary"] == "Not specified", f"{sheet} DeepASDPred stage not corrected")

    rq3_summary = pd.read_csv(BASE_DIR / "rq3_results_q1_v2" / "rq3_global_practice_summary_q1_v2.csv")
    ext = rq3_summary[rq3_summary["practice_signal"].eq("q3_external_validation_signal")].iloc[0]
    require(int(ext["positive_n"]) == 15 and int(ext["total_n"]) - int(ext["positive_n"]) == 261, "RQ3 partition is not 261+15=276")

    rq2_denoms = pd.read_csv(BASE_DIR / "rq2_results_q1_v2" / "rq2_denominators_q1_v2.csv")
    require(set(rq2_denoms["subset"]) == {"all_rows", "title_abstract_signal_state"}, "RQ2 denominator is not labeled as title/abstract")
    caption = (BASE_DIR / "rq2_results_q1_v2" / "rq2_heatmap_and_timing_q1_v2_caption.txt").read_text(encoding="utf-8")
    require("not be reported as confirmed clinical integration" in caption, "RQ2 caption does not warn against confirmed integration")
    require(not (BASE_DIR / "rq2_results_q1_v2" / "rq2_stage_x_integration_q1_v2.csv").exists(), "Stale RQ2 integration CSV still exists")

    not_spec = pd.read_csv(BASE_DIR / "rq1_results_q1_v2" / "rq1_denominators_q1_v2.csv")
    require(int(not_spec.loc[not_spec["group"].eq("Not specified"), "denominator_n"].iloc[0]) == 8, "RQ1 Not specified count is not 8")

    require(int(prisma.loc[prisma["stage"].eq("Combined provisional candidate pool"), "n"].iloc[0]) == 468, "PRISMA pool is not 468")

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

    print("PASS: dataset, counts, binary columns, traceability-sensitive corrections, and figures validated.")


if __name__ == "__main__":
    main()
