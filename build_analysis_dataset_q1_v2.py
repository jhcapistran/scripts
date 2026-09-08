from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MASTER_FILE = BASE_DIR / "cribado_maestro_276_actualizacion_FINAL_CORREGIDO_2026-09-07.xlsx"
OUTPUT_FILE = BASE_DIR / "analysis_dataset_q1_v2.xlsx"

HISTORICAL_RQ12_SHEET = "RQ1_RQ2_base_276"
HISTORICAL_RQ3_SHEET = "RQ3_base_276"
NEW_EXTRACTION_SHEET = "Extraccion_RQ_nuevos"
TEXT_COMPLETE_SHEET = "Texto_completo_191"

STAGE_FLAG_COLS = {
    "stage_prescreening": "Prescreening",
    "stage_screening": "Screening",
    "stage_diagnosis": "Diagnosis",
    "stage_prognosis": "Prognosis",
    "stage_monitoring_intervention": "Monitoring/intervention",
}

BINARY_COLS = [
    "ds_neuroimaging",
    "ds_physiological",
    "ds_behavioral_video",
    "ds_voice_audio",
    "ds_structured_records",
    "ds_other_biological",
    "ds_other",
    *STAGE_FLAG_COLS,
    "q2_candidate_abstract",
    "q2_candidate_terms",
    "q3_external_validation_signal",
    "q3_explainability_signal",
    "q3_multisite_signal",
    "q3_multisource_strategy_signal",
    "needs_full_text_check",
    "q3_completed_manual_19",
    "q3_external_validation",
    "q3_multisite_dataset",
    "q3_cross_site_robustness",
    "q3_prospective_evaluation",
    "q3_xai_strict",
    "q3_xai_partial",
    "q3_multisource_data",
]

NEW_TO_EXISTING = {
    "DOI": "doi",
    "data_source_primary": "modalidad",
    "ai_type": "tipo_IA",
    "ai_algorithm_main": "AI_algorithm_main",
    "ai_task_type": "AI_task_type",
}


def clean_text(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    return text if text else pd.NA


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
        raise ValueError(f"Cannot coerce non-binary value {value!r}")
    return int(float(value) > 0)


def restore_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in BINARY_COLS:
        if col in work.columns:
            work[col] = work[col].map(binary_value).astype("int64")
    if "q3_candidate_terms" in work.columns:
        work["q3_candidate_terms"] = work["q3_candidate_terms"].map(clean_text)
    return work


def norm_key(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip().casefold())


def modality_bucket(value: object) -> str:
    text = norm_key(value)
    if not text:
        return "Not specified"
    if any(token in text for token in ["multimodal", "multi-dataset", "multisource", "multiple public", "mixed asd"]):
        return "Multimodal"
    if any(token in text for token in ["omics", "proteomics", "transcript", "methylation", "genetic", "snp", "microbiome", "biomarker", "serum", "blood", "pathology"]):
        return "Biological/omics"
    if any(token in text for token in ["eeg", "fnirs", "physiological", "polysomnography", "pupillary", "abr", "heart-rate", "thermal"]):
        return "Physiological signals"
    if any(token in text for token in ["nlp", "text", "ehr", "clinical reports", "language", "conversational"]):
        return "Text / NLP"
    if any(token in text for token in ["audio", "voice", "vocal", "speech"]):
        return "Audio / Voice"
    if any(token in text for token in ["image", "facial", "video", "mri", "fmri", "rs-fmri", "neuroimaging", "eye-tracking", "scanpath", "gaze", "histopathology", "drawing"]):
        return "Image"
    if any(token in text for token in ["structured", "questionnaire", "survey", "assessment", "claims", "tabular", "screening data", "demographic", "cognitive task", "game", "robot", "wearable", "platform", "interaction"]):
        return "Text / NLP"
    if "not specified" in text:
        return "Not specified"
    return "Not specified"


def add_traceability(df: pd.DataFrame, source: str) -> pd.DataFrame:
    work = df.copy()
    reviewer = work.get("reviewer", work.get("coder", work.get("assigned_to", pd.Series([pd.NA] * len(work)))))
    work["reviewer_trace"] = reviewer.fillna("Not recorded")
    for col in ["reviewer_1", "reviewer_2", "adjudicator"]:
        if col not in work.columns:
            work[col] = pd.NA
    if "decision_date" not in work.columns:
        work["decision_date"] = pd.NA
    work["decision_date_trace"] = work["decision_date"].fillna("Not recorded")
    work["traceability_source"] = source
    return work


def harmonize_historical(df: pd.DataFrame, source_sheet: str) -> pd.DataFrame:
    work = df.drop(columns=["paper_order"], errors="ignore").copy()
    work["cohort"] = "historical_276"
    work["source_record_id"] = work["study_id"]
    work["full_text_decision"] = "Historical include"
    work["rq2_integration_status"] = pd.NA
    work["rq2_role"] = pd.NA
    if "q3_external_validation" not in work.columns:
        work["q3_external_validation"] = work.get("q3_external_validation_signal", 0)
    for col in ["q3_multisite_dataset", "q3_cross_site_robustness", "q3_prospective_evaluation", "q3_xai_strict", "q3_xai_partial", "q3_multisource_data"]:
        if col not in work.columns:
            work[col] = 0
    return restore_binary_columns(add_traceability(work, f"{source_sheet} copied from master"))


def title_column(df: pd.DataFrame) -> str:
    matches = [col for col in df.columns if "t" in col.casefold() and "tulo" in col.casefold()]
    if not matches:
        raise ValueError("Could not locate title column in new extraction sheet.")
    return matches[0]


def year_column(df: pd.DataFrame) -> str:
    matches = [col for col in df.columns if col != "bib_index" and col.casefold().startswith("a")]
    if not matches:
        raise ValueError("Could not locate year column in text-complete sheet.")
    return matches[0]


def harmonize_new(extraction: pd.DataFrame, text_complete: pd.DataFrame, template_columns: list[str], first_paper_order: int) -> pd.DataFrame:
    included = extraction[extraction["full_text_decision"].isin(["Include", "Include with integrity flag"])].copy()
    years = text_complete.set_index("bib_index")[year_column(text_complete)]
    work = pd.DataFrame(index=included.index)
    for old, new in NEW_TO_EXISTING.items():
        work[new] = included[old].map(clean_text)
    work["title"] = included[title_column(included)].map(clean_text)
    work["year"] = included["bib_index"].map(years).map(clean_text)
    work["abstract"] = included["evidence_summary"].map(clean_text)
    work["assigned_to"] = included["reviewer"].map(clean_text)
    work["coder"] = included["reviewer"].map(clean_text)
    work["study_id"] = included["bib_index"].map(lambda value: f"new_{int(value)}")
    work["source_record_id"] = included["bib_index"]
    work["cohort"] = "update_2026_included"
    work["modalidad_raw"] = work["modalidad"]
    work["modalidad"] = work["modalidad_raw"].map(modality_bucket)
    work["stage_primary"] = included["stage_primary"].map(clean_text)
    work["confidence"] = pd.NA
    work["learning_paradigm"] = pd.NA
    work["source_selected"] = included["source_url"].map(clean_text)
    work["source_rule"] = "Extraccion_RQ_nuevos adjudicated full-text extraction"
    work["notes_coding"] = included["evidence_summary"].map(clean_text)
    work["integrity_status"] = included["integrity_status"].map(clean_text)
    work["full_text_decision"] = included["full_text_decision"].map(clean_text)
    work["reviewer"] = included["reviewer"].map(clean_text)
    work["decision_date"] = included["decision_date"]
    work["rq2_integration_status"] = included["rq2_integration_status"].map(clean_text)
    work["rq2_role"] = included["rq2_role"].map(clean_text)
    for col in ["q3_external_validation", "q3_multisite_dataset", "q3_cross_site_robustness", "q3_prospective_evaluation", "q3_xai_strict", "q3_xai_partial", "q3_multisource_data"]:
        work[col] = included[col]
    work["q3_external_validation_signal"] = work["q3_external_validation"]
    work["q3_explainability_signal"] = work[["q3_xai_strict", "q3_xai_partial"]].max(axis=1)
    work["q3_multisite_signal"] = work[["q3_multisite_dataset", "q3_cross_site_robustness"]].max(axis=1)
    work["q3_multisource_strategy_signal"] = work["q3_multisource_data"]
    work["q3_candidate_terms"] = pd.NA
    work["q2_candidate_abstract"] = pd.NA
    work["q2_candidate_terms"] = pd.NA
    work["needs_full_text_check"] = 0
    work["q3_completed_manual_19"] = 1
    for col in ["ds_neuroimaging", "ds_physiological", "ds_behavioral_video", "ds_voice_audio", "ds_structured_records", "ds_other_biological", "ds_other"]:
        work[col] = 0
    for col, label in STAGE_FLAG_COLS.items():
        work[col] = work["stage_primary"].eq(label).astype(int)
    work["n_data_sources"] = pd.NA
    work["ds_other_notes"] = pd.NA
    for col in template_columns:
        if col not in work.columns:
            work[col] = pd.NA
    work = work[template_columns + [col for col in work.columns if col not in template_columns]]
    work.insert(0, "paper_order", range(first_paper_order, first_paper_order + len(work)))
    return restore_binary_columns(add_traceability(work, f"{NEW_EXTRACTION_SHEET} included full-text decisions"))


def add_paper_order(df: pd.DataFrame) -> pd.DataFrame:
    work = df.drop(columns=["paper_order"], errors="ignore").copy()
    work.insert(0, "paper_order", range(1, len(work) + 1))
    return work


def require_unique(df: pd.DataFrame) -> None:
    if df["study_id"].nunique() != len(df):
        raise ValueError("Combined corpus has duplicated study_id values.")
    for col in ["doi", "title"]:
        keys = df[col].map(norm_key)
        duplicated = keys[keys.ne("") & keys.duplicated(keep=False)]
        if not duplicated.empty:
            raise ValueError(f"Combined corpus has duplicated {col} values; resolve in master instead of inferring.")


def build_readme(total_n: int, historical_n: int, new_n: int, excluded_n: int, pending_n: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"item": "Master workbook", "value": MASTER_FILE.name, "note": "Only input workbook used by this repository."},
            {"item": "Final analytical corpus", "value": total_n, "note": f"{historical_n} historical included studies plus {new_n} newly included full-text studies."},
            {"item": "Historical source sheets", "value": f"{HISTORICAL_RQ12_SHEET}; {HISTORICAL_RQ3_SHEET}", "note": "Copied from the adjudicated historical base."},
            {"item": "New included source sheet", "value": NEW_EXTRACTION_SHEET, "note": "Only Include and Include with integrity flag rows are appended."},
            {"item": "Full-text exclusions", "value": excluded_n, "note": f"Counted from {TEXT_COMPLETE_SHEET}.full_text_decision == Exclude."},
            {"item": "Pending full-text adjudications", "value": pending_n, "note": "No eligibility is recalculated by scripts."},
            {"item": "RQ2 policy", "value": "rq2_integration_status/rq2_role only", "note": "Legacy q2_candidate_* columns are retained for audit only and are not treated as final clinical integration."},
            {"item": "Unmapped without inference", "value": "historical RQ2 final integration", "note": "The 276 historical rows do not contain rq2_integration_status or rq2_role in the master."},
        ]
    )


def build_prisma_scope(total_n: int, historical_n: int, new_n: int, excluded_n: int, pending_n: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"stage": "Historical included studies", "n": historical_n, "note": "From adjudicated historical base sheets."},
            {"stage": "New full-text included studies", "n": new_n, "note": "From Extraccion_RQ_nuevos Include / Include with integrity flag."},
            {"stage": "Full-text exclusions", "n": excluded_n, "note": "From Texto_completo_191 full_text_decision == Exclude."},
            {"stage": "Pending full-text adjudication", "n": pending_n, "note": "Must remain zero for the corrected master."},
            {"stage": "Final analytical corpus", "n": total_n, "note": "Unique studies evaluated by RQ1/RQ2/RQ3 scripts."},
        ]
    )


def build_included_studies(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "paper_order",
        "study_id",
        "cohort",
        "source_record_id",
        "year",
        "title",
        "doi",
        "assigned_to",
        "modalidad",
        "tipo_IA",
        "AI_algorithm_main",
        "AI_task_type",
        "stage_primary",
        "rq2_integration_status",
        "rq2_role",
        "full_text_decision",
    ]
    return df[[col for col in columns if col in df.columns]].copy()


def main() -> None:
    historical_rq12 = harmonize_historical(pd.read_excel(MASTER_FILE, sheet_name=HISTORICAL_RQ12_SHEET), HISTORICAL_RQ12_SHEET)
    historical_rq3 = harmonize_historical(pd.read_excel(MASTER_FILE, sheet_name=HISTORICAL_RQ3_SHEET), HISTORICAL_RQ3_SHEET)
    if historical_rq12["study_id"].tolist() != historical_rq3["study_id"].tolist():
        raise ValueError("Historical RQ1/RQ2 and RQ3 study order differs.")

    extraction = pd.read_excel(MASTER_FILE, sheet_name=NEW_EXTRACTION_SHEET)
    text_complete = pd.read_excel(MASTER_FILE, sheet_name=TEXT_COMPLETE_SHEET)
    new_rq12 = harmonize_new(extraction, text_complete, historical_rq12.drop(columns=["paper_order"], errors="ignore").columns.tolist(), len(historical_rq12) + 1)
    new_rq3 = harmonize_new(extraction, text_complete, historical_rq3.drop(columns=["paper_order"], errors="ignore").columns.tolist(), len(historical_rq3) + 1)

    rq12 = add_paper_order(pd.concat([historical_rq12, new_rq12], ignore_index=True))
    rq3 = add_paper_order(pd.concat([historical_rq3, new_rq3], ignore_index=True))
    require_unique(rq12)
    require_unique(rq3)

    historical_n = len(historical_rq12)
    new_n = len(new_rq12)
    total_n = len(rq12)
    excluded_n = int(text_complete["full_text_decision"].eq("Exclude").sum())
    pending_n = int(text_complete["full_text_decision"].astype(str).str.contains("Pending", case=False, na=False).sum())
    if total_n != historical_n + new_n:
        raise ValueError("Final corpus count does not equal historical + new included studies.")

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        build_readme(total_n, historical_n, new_n, excluded_n, pending_n).to_excel(writer, sheet_name="README_FINAL", index=False)
        build_prisma_scope(total_n, historical_n, new_n, excluded_n, pending_n).to_excel(writer, sheet_name="PRISMA_scope", index=False)
        build_included_studies(rq12).to_excel(writer, sheet_name="included_studies", index=False)
        new_rq12.to_excel(writer, sheet_name="new_included_studies", index=False)
        rq12.to_excel(writer, sheet_name="rq1_rq2_graph_ready", index=False)
        rq3.to_excel(writer, sheet_name="rq3_graph_ready", index=False)

    print(f"Wrote {OUTPUT_FILE.name}: {total_n} unique studies = {historical_n} historical + {new_n} new; full-text exclusions={excluded_n}; pending={pending_n}.")


if __name__ == "__main__":
    main()
