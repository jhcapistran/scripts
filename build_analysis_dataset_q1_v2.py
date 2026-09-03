from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MASTER_FILE = BASE_DIR / "cribado_maestro_276_actualizacion_2026-09-02.xlsx"
OUTPUT_FILE = BASE_DIR / "analysis_dataset_q1_v2.xlsx"
ANALYTICAL_N = 276
PROVISIONAL_NEW_N = 192
PROVISIONAL_POOL_N = ANALYTICAL_N + PROVISIONAL_NEW_N
MASTER_DATE = "2026-09-02"
CORRECTION_DATE = "2026-09-03"

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
TEXT_COLS = ["q3_candidate_terms"]


def add_paper_order(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if work["study_id"].nunique() != ANALYTICAL_N or len(work) != ANALYTICAL_N:
        raise ValueError(f"Expected {ANALYTICAL_N} unique evaluated studies, found rows={len(work)} unique={work['study_id'].nunique()}.")
    work.insert(0, "paper_order", range(1, len(work) + 1))
    return work


def binary_value(value: object) -> int:
    if pd.isna(value):
        return 0
    if isinstance(value, dt.datetime):
        return int(value.date() == dt.date(1900, 1, 1))
    if isinstance(value, dt.time):
        return 0
    if isinstance(value, str):
        text = value.strip().casefold()
        if text in {"1", "1.0", "true", "yes", "y", "si", "s"}:
            return 1
        if text in {"0", "0.0", "false", "no", "n", ""}:
            return 0
        return value
    return int(float(value) > 0)


def restore_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in BINARY_COLS:
        if col not in work.columns:
            continue
        converted = work[col].map(binary_value)
        if converted.map(lambda value: isinstance(value, str)).any():
            continue
        work[col] = converted.astype("int64")
    for col in TEXT_COLS:
        if col in work.columns:
            work[col] = work[col].map(lambda value: pd.NA if pd.isna(value) else str(value))
    return work


def add_traceability(df: pd.DataFrame, source: str) -> pd.DataFrame:
    work = df.copy()
    reviewer = work.get("coder", work.get("assigned_to", pd.Series(["Not recorded"] * len(work))))
    work["reviewer_trace"] = reviewer.fillna("Not recorded")
    for col in ["reviewer_1", "reviewer_2", "adjudicator", "decision_date"]:
        if col not in work.columns:
            work[col] = pd.NA
    if "assigned_to" in work.columns:
        work["reviewer_assignment_trace"] = work["assigned_to"].fillna("Not recorded")
    work["decision_date_trace"] = "Not recorded"
    work["traceability_source"] = source
    return work


def prepare_update_sheet(update: pd.DataFrame, full_text: pd.DataFrame) -> pd.DataFrame:
    work = update.copy()
    for col in ["reviewer_1", "reviewer_2", "adjudicator", "decision_date"]:
        if col not in work.columns:
            work[col] = pd.NA
    text_cols = [
        "bib_index",
        "full_text_status",
        "integrity_status",
        "full_text_decision",
        "final_exclusion_reason",
        "reviewer",
        "decision_date",
        "reviewer_1",
        "reviewer_2",
        "adjudicator",
    ]
    text = full_text[[col for col in text_cols if col in full_text.columns]].copy()
    merged = work.merge(text, on="bib_index", how="left", suffixes=("", "_text"))
    text_backed = merged["eligibility_bucket"].eq("Provisional include")
    for col in [c for c in text_cols if c != "bib_index" and c in work.columns and f"{c}_text" in merged.columns]:
        merged.loc[text_backed, col] = merged.loc[text_backed, f"{col}_text"].combine_first(merged.loc[text_backed, col])
        merged = merged.drop(columns=[f"{col}_text"])
    work = merged
    work["reviewer_trace"] = work.get("reviewer", pd.Series([pd.NA] * len(work))).fillna("Not recorded")
    work["decision_date_trace"] = work.get("decision_date", pd.Series([pd.NA] * len(work))).fillna("Not recorded")
    work["traceability_source"] = f"{MASTER_FILE.name}:Actualizacion_430"
    return work


def build_readme() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item": "Analytical universe",
                "value": ANALYTICAL_N,
                "note": "These are the studies evaluated in the final graph-ready corpus.",
            },
            {
                "item": "Provisional candidate pool",
                "value": PROVISIONAL_POOL_N,
                "note": "276 prior included studies plus 192 new candidates that passed title/abstract screening, including bib_index 188.",
            },
            {
                "item": "Important distinction",
                "value": "468 is not final included N",
                "note": "The 192 new candidates must be counted in screening/PRISMA flow, but no new full-text inclusion decision is inferred here.",
            },
            {
                "item": "Master workbook",
                "value": MASTER_FILE.name,
                "note": f"Current master file dated {MASTER_DATE}; Base_276 is preserved as the final analytical corpus.",
            },
            {
                "item": "Use for PRISMA/paper ordering",
                "value": "paper_order",
                "note": "Sequential order from 1 to 276. Keep study_id as the original traceability identifier.",
            },
            {
                "item": "study_id",
                "value": "Original identifier",
                "note": "study_id can be non-sequential; it is not the evaluated-study count.",
            },
            {
                "item": "Historical/audit workbooks",
                "value": "Not final denominators",
                "note": "Older source and audit sheets may contain historical rows. Use this workbook for final analysis tables.",
            },
            {
                "item": "RQ1/RQ2 source",
                "value": MASTER_FILE.name,
                "note": "Sheet: RQ1_RQ2_base_276.",
            },
            {
                "item": "RQ3 source",
                "value": MASTER_FILE.name,
                "note": "Sheet: RQ3_base_276.",
            },
        ]
    )


def build_prisma_scope() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stage": "Combined provisional candidate pool",
                "n": PROVISIONAL_POOL_N,
                "note": "276 final prior studies plus 192 new candidates that passed screening; not a final included-study denominator.",
            },
            {
                "stage": "New candidates passed title/abstract screening",
                "n": PROVISIONAL_NEW_N,
                "note": "These passed the update screening and must be counted in screening/PRISMA summaries; includes bib_index 188 as a candidate.",
            },
            {
                "stage": "Final analytical corpus evaluated",
                "n": ANALYTICAL_N,
                "note": "Final graph-ready studies evaluated for the review questions in this repository.",
            },
            {
                "stage": "RQ1 evaluated",
                "n": ANALYTICAL_N,
                "note": "Same analytical corpus; no additional exclusion step is applied by RQ1 script.",
            },
            {
                "stage": "RQ2 evaluated",
                "n": ANALYTICAL_N,
                "note": "Same analytical corpus; q2 signal states are derived within these 276 studies.",
            },
            {
                "stage": "RQ3 evaluated",
                "n": ANALYTICAL_N,
                "note": "Same analytical corpus; Q3 practice signals are derived within these 276 studies.",
            },
        ]
    )


def build_included_studies(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "paper_order",
        "study_id",
        "year",
        "title",
        "doi",
        "assigned_to",
        "modalidad",
        "tipo_IA",
        "stage_primary",
    ]
    return df[[col for col in columns if col in df.columns]].copy()


def main() -> None:
    rq12 = pd.read_excel(MASTER_FILE, sheet_name="RQ1_RQ2_base_276").drop(columns=["paper_order"], errors="ignore")
    rq3 = pd.read_excel(MASTER_FILE, sheet_name="RQ3_base_276").drop(columns=["paper_order"], errors="ignore")
    rq12 = add_traceability(restore_binary_columns(add_paper_order(rq12)), "RQ1_RQ2_base_276 copied from master")
    rq3 = add_traceability(restore_binary_columns(add_paper_order(rq3)), "RQ3_base_276 copied from master")
    if rq12["study_id"].tolist() != rq3["study_id"].tolist():
        raise ValueError("RQ1/RQ2 and RQ3 graph-ready study order differs.")
    update = prepare_update_sheet(
        pd.read_excel(MASTER_FILE, sheet_name="Actualizacion_430"),
        pd.read_excel(MASTER_FILE, sheet_name="Texto_completo_192"),
    )
    passed = update[update["eligibility_bucket"].eq("Provisional include")].copy()
    if len(passed) != PROVISIONAL_NEW_N:
        raise ValueError(f"Expected {PROVISIONAL_NEW_N} provisional new candidates, found {len(passed)}.")

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        build_readme().to_excel(writer, sheet_name="README_FINAL", index=False)
        build_prisma_scope().to_excel(writer, sheet_name="PRISMA_scope_468", index=False)
        build_included_studies(rq12).to_excel(writer, sheet_name="included_studies_276", index=False)
        passed.to_excel(writer, sheet_name="new_candidates_passed_192", index=False)
        rq12.to_excel(writer, sheet_name="rq1_rq2_graph_ready", index=False)
        rq3.to_excel(writer, sheet_name="rq3_graph_ready", index=False)

    print(f"Wrote {OUTPUT_FILE.name} with {ANALYTICAL_N} evaluated studies and {PROVISIONAL_POOL_N} provisional candidates counted.")


if __name__ == "__main__":
    main()
