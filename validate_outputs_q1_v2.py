from __future__ import annotations

from pathlib import Path

from openpyxl import load_workbook
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MASTER_FILE = BASE_DIR / "cribado_maestro_276_actualizacion_2026-09-02.xlsx"
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
    require(MASTER_FILE.exists(), "master workbook does not exist")
    require(DATASET.exists(), "analysis_dataset_q1_v2.xlsx does not exist")

    master_sheets = pd.ExcelFile(MASTER_FILE).sheet_names
    require("Texto_completo_192" in master_sheets, "Master is missing Texto_completo_192")
    require("Texto_completo_191" not in master_sheets, "Stale Texto_completo_191 sheet still exists")
    require("Excluidos_174" in master_sheets, "Master is missing Excluidos_174")
    master_update = pd.read_excel(MASTER_FILE, sheet_name="Actualizacion_430")
    require(int(master_update["eligibility_bucket"].eq("Provisional include").sum()) == 192, "Master does not have 192 provisional candidates")
    require(int(master_update["eligibility_bucket"].eq("Excluded").sum()) == 174, "Master does not have 174 excluded records")
    require(int(master_update["eligibility_bucket"].eq("Report not retrieved").sum()) == 0, "Master reports nonzero reports not retrieved")
    master_text = pd.read_excel(MASTER_FILE, sheet_name="Texto_completo_192")
    require(len(master_text) == 192, "Texto_completo_192 does not contain 192 rows")
    require(int(master_text["full_text_decision"].eq("Pending adjudication").sum()) == 191, "Master does not have 191 pending candidates")
    wb = load_workbook(MASTER_FILE, data_only=False)
    require("Texto_completo_192" in str(wb["Resumen"]["B11"].value), "Resumen formulas do not reference Texto_completo_192")
    require("$J$2:$J$193" in str(wb["Resumen"]["B12"].value), "Resumen pending formula does not cover 192 text-complete rows")

    rq12 = read("rq1_rq2_graph_ready")
    rq3 = read("rq3_graph_ready")
    candidates = read("new_candidates_passed_192")
    prisma = read("PRISMA_scope_468")
    dataset_wb = load_workbook(DATASET, data_only=True)

    require(len(rq12) == 276 and rq12["study_id"].nunique() == 276, "RQ1/RQ2 corpus is not 276 unique studies")
    require(len(rq3) == 276 and rq3["study_id"].nunique() == 276, "RQ3 corpus is not 276 unique studies")
    require(rq12["study_id"].tolist() == rq3["study_id"].tolist(), "RQ1/RQ2 and RQ3 study order differs")

    require(len(candidates) == 192, "Expected 192 provisional candidates")
    row188 = candidates[candidates["bib_index"].eq(188)]
    require(len(row188) == 1, "bib_index 188 is not present exactly once among candidates")
    require(row188["full_text_decision"].iloc[0] == "Pending adjudication", "bib_index 188 has an inferred full-text decision")

    for df, sheet in [(rq12, "rq1_rq2_graph_ready"), (rq3, "rq3_graph_ready")]:
        check_binary_columns(df, sheet)
        for col in ["reviewer_1", "reviewer_2", "adjudicator", "decision_date"]:
            require(col in df.columns, f"{sheet} is missing traceability field {col}")
        require("q3_candidate_terms" in df.columns, f"{sheet} is missing q3_candidate_terms")
        ws = dataset_wb[sheet]
        q3_terms_col = {cell.value: cell.column for cell in ws[1]}["q3_candidate_terms"]
        require(
            all(ws.cell(row, q3_terms_col).data_type in {"s", "inlineStr", "n"} and ws.cell(row, q3_terms_col).value not in {0, 1} for row in range(2, ws.max_row + 1)),
            f"{sheet}.q3_candidate_terms was converted to binary values",
        )
        deep = df[df["title"].astype(str).str.contains("DeepASDPred", case=False, na=False)]
        require(len(deep) == 1, f"{sheet} does not contain exactly one DeepASDPred row")
        deep = deep.iloc[0]
        require(deep["modalidad"] == "Biological/omics", f"{sheet} DeepASDPred modality not corrected")
        require(deep["AI_task_type"] == "risk-RNA identification", f"{sheet} DeepASDPred task not corrected")
        require(deep["stage_primary"] == "Not specified", f"{sheet} DeepASDPred stage not corrected")
        for col in ["stage_prescreening", "stage_screening", "stage_diagnosis", "stage_prognosis", "stage_monitoring_intervention"]:
            require(int(deep[col]) == 0, f"{sheet} DeepASDPred {col} not zero")

    rq3_summary = pd.read_csv(BASE_DIR / "rq3_results_q1_v2" / "rq3_global_practice_summary_q1_v2.csv")
    ext = rq3_summary[rq3_summary["practice_signal"].eq("q3_external_validation_signal")].iloc[0]
    require(int(ext["positive_n"]) == 15 and int(ext["total_n"]) - int(ext["positive_n"]) == 261, "RQ3 external validation partition is not 261+15=276")
    q3_any = rq3[["q3_external_validation_signal", "q3_explainability_signal", "q3_multisite_signal", "q3_multisource_strategy_signal"]].astype(int).any(axis=1)
    require(int(q3_any.sum()) == 110 and int((~q3_any).sum()) == 166, "RQ3 individual signal split is not 110+166=276")
    plotted = pd.read_csv(BASE_DIR / "rq3_results_q1_v2" / "rq3_lollipop_combos_q1_v2.csv")
    omitted = pd.read_csv(BASE_DIR / "rq3_results_q1_v2" / "rq3_omitted_zero_profiles_q1_v2.csv")
    require(len(plotted) == 37 and int(plotted["combo_n"].sum()) == 262, "RQ3 plotted profiles are not 37 profiles/262 studies")
    require(len(omitted) == 10 and int(omitted["combo_n"].sum()) == 14, "RQ3 omitted profiles are not 10 profiles/14 studies")

    rq2_denoms = pd.read_csv(BASE_DIR / "rq2_results_q1_v2" / "rq2_denominators_q1_v2.csv")
    require(set(rq2_denoms["subset"]) == {"all_rows", "title_abstract_signal_state"}, "RQ2 denominator is not labeled as title/abstract")
    present = int(rq2_denoms.loc[rq2_denoms["group"].eq("Present"), "denominator_n"].iloc[0])
    absent = int(rq2_denoms.loc[rq2_denoms["group"].eq("Absent"), "denominator_n"].iloc[0])
    require(present == 242 and absent == 34, "RQ2 signal split is not 242+34=276")
    rq2_review = pd.read_csv(BASE_DIR / "rq2_results_q1_v2" / "rq2_preliminary_signals_for_manual_review_q1_v2.csv")
    require(len(rq2_review) == 242, "RQ2 manual-review export does not contain 242 records")
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
