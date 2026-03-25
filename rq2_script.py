"""
RQ2 tripartite integration model generator.

Builds a three-layer view of how AI systems are integrated into the
clinical workflow for autism spectrum disorder (ASD):
- Clinical functional stage
- Integration approach
- Decision timing

Inputs are reviewer-specific coded Excel files. Each file is filtered to the
corresponding reviewer in `assigned_to`, then all selected rows are combined.

Outputs:
- rq2_tripartite_integration_model.png
- rq2_stage_x_approach.csv
- rq2_approach_x_timing.csv
- rq2_tripartite_dataset.csv
- rq2_review_rows.csv
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib import patches
import pandas as pd
import scienceplots


# =========================================================
# CONFIGURATION
# =========================================================
INPUT_FILES = [
    {"path": "aceptados_300_v2_codificado_RA.xlsx", "reviewer": "Revisor A"},
    {"path": "aceptados_300_codificado_RB.xlsx", "reviewer": "Revisor B"},
]
SHEET_NAME = 0
OUTPUT_DIR = "rq2_results"

STAGE_COL = "stage_primary"
Q2_ABSTRACT_COL = "q2_candidate_abstract"
Q2_TERMS_COL = "q2_candidate_terms"
AI_TASK_COL = "AI_task_type"
TITLE_COL = "title"
ABSTRACT_COL = "abstract"
NOTES_COL = "notes_coding"

STUDY_ID_COL = "study_id"
YEAR_COL = "year"
DOI_COL = "doi"
MODALITY_COL = "modalidad"
# =========================================================

plt.style.use(["science", "no-latex"])


def norm_text(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().lower()
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_bool_signal(x: object) -> bool | None:
    if pd.isna(x):
        return None
    s = norm_text(x)
    if s in {"verdadero", "true", "1", "yes", "y"}:
        return True
    if s in {"falso", "false", "0", "no", "n"}:
        return False
    return None


def normalize_stage(x: object) -> str:
    s = norm_text(x)
    if s in {"", "na", "n/a", "none", "null"}:
        return "Not specified"
    if "not spec" in s or "no especific" in s or "unclear" in s or "not clear" in s:
        return "Not specified"
    if "prescreen" in s:
        return "Prescreening"
    if "monitor" in s or "intervention" in s:
        return "Monitoring/intervention"
    if "diagnos" in s:
        return "Diagnosis"
    if "prognos" in s:
        return "Prognosis"
    if "screen" in s:
        return "Screening"
    return "Not specified"


def load_reviewer_inputs() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for spec in INPUT_FILES:
        path = spec["path"]
        reviewer = spec["reviewer"]

        df_or_sheets = pd.read_excel(path, sheet_name=SHEET_NAME)
        if isinstance(df_or_sheets, dict):
            if not df_or_sheets:
                raise ValueError(f"Excel file has no sheets: {path}")
            df_part = next(iter(df_or_sheets.values()))
        else:
            df_part = df_or_sheets

        if "assigned_to" not in df_part.columns:
            raise ValueError(f"Missing required column 'assigned_to' in: {path}")

        filtered = df_part[df_part["assigned_to"].astype(str).str.strip() == reviewer].copy()
        filtered["source_file"] = path
        frames.append(filtered)

    if not frames:
        raise ValueError("No input data was loaded.")

    return pd.concat(frames, ignore_index=True)


def derive_integration_approach(row: pd.Series) -> str:
    stage = row["stage_norm"]
    q2_abs = row["q2_abstract_bool"]
    q2_terms = row["q2_terms_bool"]

    title = row["title_norm"]
    abstract = row["abstract_norm"]
    ai_task = row["ai_task_norm"]
    notes = row["notes_norm"]
    text = " ".join([title, abstract, ai_task, notes])

    if q2_abs is False and q2_terms is False:
        return "No clear integration"

    if any(k in text for k in [
        "questionnaire", "checklist", "triage", "referral prioritization",
        "refer", "pre-screen", "prescreen", "screening form"
    ]):
        return "Triage / questionnaires"

    if any(k in text for k in [
        "mobile", "smartphone", "app-based", "m-health", "mhealth", "tablet-based"
    ]):
        return "Mobile screening"

    if any(k in text for k in [
        "feature extraction", "feature selection", "marker extraction",
        "biomarker extraction", "representation learning", "signal feature"
    ]):
        return "Feature extraction"

    if any(k in text for k in [
        "second reader", "second-reader", "decision support", "computer-aided",
        "computer aided", "reader support", "clinician support", "classification"
    ]):
        return "Second-reader decision support"

    if stage == "Prognosis" or any(k in text for k in [
        "risk stratification", "risk", "predictor", "prediction model",
        "prenatal", "perinatal", "maternal"
    ]):
        return "Risk stratification"

    if any(k in text for k in [
        "dashboard", "longitudinal", "long-term follow-up", "follow-up",
        "trajectory", "progress tracking", "monitoring dashboard"
    ]):
        return "Longitudinal dashboards"

    if any(k in text for k in [
        "intervention", "therapy", "adaptive", "personalized", "personalisation",
        "personalization", "robot-assisted", "serious game", "virtual reality"
    ]):
        return "Adaptive intervention"

    if any(k in text for k in [
        "assistive", "educational", "education", "school", "communication aid",
        "support tool", "caregiver support"
    ]):
        return "Assistive tools"

    if stage in {"Prescreening", "Screening"}:
        return "Triage / questionnaires"
    if stage == "Diagnosis":
        return "Second-reader decision support"
    if stage == "Monitoring/intervention":
        return "Adaptive intervention"

    if q2_abs is True or q2_terms is True:
        return "Unspecified integration"

    return "Not specified"


def explain_integration_approach(row: pd.Series) -> str:
    stage = row["stage_norm"]
    q2_abs = row["q2_abstract_bool"]
    q2_terms = row["q2_terms_bool"]
    text = " ".join([row["title_norm"], row["abstract_norm"], row["ai_task_norm"], row["notes_norm"]])

    if q2_abs is False and q2_terms is False:
        return "Both q2 candidate signals are explicitly false"
    if any(k in text for k in [
        "questionnaire", "checklist", "triage", "referral prioritization",
        "refer", "pre-screen", "prescreen", "screening form"
    ]):
        return "Keyword rule matched triage/questionnaire workflow"
    if any(k in text for k in [
        "mobile", "smartphone", "app-based", "m-health", "mhealth", "tablet-based"
    ]):
        return "Keyword rule matched mobile screening workflow"
    if any(k in text for k in [
        "feature extraction", "feature selection", "marker extraction",
        "biomarker extraction", "representation learning", "signal feature"
    ]):
        return "Keyword rule matched feature extraction workflow"
    if any(k in text for k in [
        "second reader", "second-reader", "decision support", "computer-aided",
        "computer aided", "reader support", "clinician support", "classification"
    ]):
        return "Keyword rule matched second-reader decision support"
    if stage == "Prognosis" or any(k in text for k in [
        "risk stratification", "risk", "predictor", "prediction model",
        "prenatal", "perinatal", "maternal"
    ]):
        return "Stage or keywords matched risk stratification"
    if any(k in text for k in [
        "dashboard", "longitudinal", "long-term follow-up", "follow-up",
        "trajectory", "progress tracking", "monitoring dashboard"
    ]):
        return "Keyword rule matched longitudinal dashboard workflow"
    if any(k in text for k in [
        "intervention", "therapy", "adaptive", "personalized", "personalisation",
        "personalization", "robot-assisted", "serious game", "virtual reality"
    ]):
        return "Keyword rule matched adaptive intervention workflow"
    if any(k in text for k in [
        "assistive", "educational", "education", "school", "communication aid",
        "support tool", "caregiver support"
    ]):
        return "Keyword rule matched assistive tool workflow"
    if stage in {"Prescreening", "Screening"}:
        return "Fallback assigned from prescreening/screening stage"
    if stage == "Diagnosis":
        return "Fallback assigned from diagnosis stage"
    if stage == "Monitoring/intervention":
        return "Fallback assigned from monitoring/intervention stage"
    if q2_abs is True or q2_terms is True:
        return "Q2 signal is positive but no specific integration workflow was matched"
    return "No keyword or stage fallback matched; kept as Not specified"


def derive_decision_timing(row: pd.Series) -> str:
    approach = row["integration_approach"]
    stage = row["stage_norm"]
    text = " ".join([row["title_norm"], row["abstract_norm"], row["ai_task_norm"], row["notes_norm"]])

    if approach in {"Triage / questionnaires", "Mobile screening", "Feature extraction", "Risk stratification"}:
        return "Pre-decision"

    if approach == "Second-reader decision support":
        if any(k in text for k in ["triage", "prioritization", "pre-read", "pre read", "feature extraction"]):
            return "Pre-decision"
        return "In-decision"

    if approach in {"Longitudinal dashboards", "Adaptive intervention", "Assistive tools"}:
        return "Post-decision"

    if approach == "No clear integration":
        return "Unspecified"

    if stage in {"Prescreening", "Screening", "Prognosis"}:
        return "Pre-decision"
    if stage == "Diagnosis":
        return "In-decision"
    if stage == "Monitoring/intervention":
        return "Post-decision"

    return "Unspecified"


def explain_decision_timing(row: pd.Series) -> str:
    approach = row["integration_approach"]
    stage = row["stage_norm"]
    text = " ".join([row["title_norm"], row["abstract_norm"], row["ai_task_norm"], row["notes_norm"]])

    if approach in {"Triage / questionnaires", "Mobile screening", "Feature extraction", "Risk stratification"}:
        return "Approach maps directly to pre-decision support"
    if approach == "Second-reader decision support":
        if any(k in text for k in ["triage", "prioritization", "pre-read", "pre read", "feature extraction"]):
            return "Second-reader pattern was re-routed to pre-decision by text cue"
        return "Second-reader pattern maps to in-decision support"
    if approach in {"Longitudinal dashboards", "Adaptive intervention", "Assistive tools"}:
        return "Approach maps directly to post-decision support"
    if approach == "No clear integration":
        return "No clear integration cannot be placed on the decision timeline"
    if stage in {"Prescreening", "Screening", "Prognosis"}:
        return "Fallback assigned from stage to pre-decision"
    if stage == "Diagnosis":
        return "Fallback assigned from stage to in-decision"
    if stage == "Monitoring/intervention":
        return "Fallback assigned from stage to post-decision"
    return "No approach or stage fallback matched; kept as Unspecified"


def build_review_reasons(row: pd.Series) -> str:
    reasons: list[str] = []
    if row["integration_approach"] in {"Unspecified integration", "No clear integration", "Not specified"}:
        reasons.append(f"integration_approach={row['integration_approach']}")
    if row["decision_timing"] == "Unspecified":
        reasons.append("decision_timing=Unspecified")
    if row["stage_norm"] == "Not specified":
        reasons.append("stage_norm=Not specified")
    if pd.isna(row["q2_abstract_bool"]):
        reasons.append("q2_abstract_bool=missing")
    if pd.isna(row["q2_terms_bool"]):
        reasons.append("q2_terms_bool=missing")
    return "; ".join(reasons)


def wrap_label(label: str, width: int = 22) -> str:
    words = label.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines)


def compute_node_positions(labels: list[str], top: float = 0.92, bottom: float = 0.08) -> dict[str, float]:
    if len(labels) == 1:
        return {labels[0]: 0.5}
    step = (top - bottom) / (len(labels) - 1)
    return {label: top - i * step for i, label in enumerate(labels)}


def draw_tripartite_figure(df: pd.DataFrame, outpath: Path) -> None:
    stage_order = [
        "Prescreening",
        "Screening",
        "Diagnosis",
        "Prognosis",
        "Monitoring/intervention",
        "Not specified",
    ]
    approach_order = [
        "Triage / questionnaires",
        "Mobile screening",
        "Second-reader decision support",
        "Feature extraction",
        "Risk stratification",
        "Longitudinal dashboards",
        "Adaptive intervention",
        "Assistive tools",
        "Unspecified integration",
        "No clear integration",
        "Not specified",
    ]
    stage_labels = [x for x in stage_order if x in set(df["stage_norm"])]
    approach_labels = [x for x in approach_order if x in set(df["integration_approach"])]

    counts = (
        pd.crosstab(df["integration_approach"], df["stage_norm"])
        .reindex(index=approach_labels, columns=stage_labels, fill_value=0)
    )
    shares = counts.div(counts.sum(axis=0), axis=1).fillna(0)
    stage_totals = counts.sum(axis=0)
    max_count = max(int(counts.to_numpy().max()), 1)
    cmap = plt.cm.YlGnBu
    norm = plt.Normalize(vmin=0, vmax=max_count)

    fig, heat_ax = plt.subplots(figsize=(12.3, 8.2), facecolor="white")
    heat_ax.set_facecolor("white")

    for row_idx, approach in enumerate(approach_labels):
        for col_idx, stage in enumerate(stage_labels):
            value = int(counts.loc[approach, stage])
            share = shares.loc[approach, stage]
            base = cmap(norm(value)) if value else "#fbf8f1"
            rect = patches.FancyBboxPatch(
                (col_idx, row_idx),
                1,
                1,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                linewidth=1.5 if value else 0.7,
                edgecolor="#e6e6e6" if value else "#f0f0f0",
                facecolor=base if value else "white",
            )
            heat_ax.add_patch(rect)
            label = f"{value}"
            if value:
                label += f"\n{share * 100:.1f}%"
            heat_ax.text(
                col_idx + 0.5,
                row_idx + 0.5,
                label,
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold" if value == max_count else "normal",
                color="#1f2933" if norm(value) < 0.55 else "white",
            )

    heat_ax.set_xlim(0, len(stage_labels))
    heat_ax.set_ylim(len(approach_labels), 0)
    heat_ax.set_xticks([x + 0.5 for x in range(len(stage_labels))])
    heat_ax.set_xticklabels(
        [f"{wrap_label(label, 14)}\n(n={int(stage_totals[label])})" for label in stage_labels],
        fontsize=11,
    )
    heat_ax.set_yticks([y + 0.5 for y in range(len(approach_labels))])
    heat_ax.set_yticklabels([wrap_label(label, 24) for label in approach_labels], fontsize=10)
    heat_ax.tick_params(length=0)
    for spine in heat_ax.spines.values():
        spine.set_visible(False)
    heat_ax.set_xlabel("Clinical functional stage", fontsize=11, labelpad=12)
    heat_ax.set_ylabel("Integration approach", fontsize=11, labelpad=12)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=heat_ax, fraction=0.03, pad=0.02)
    cbar.set_label("Frequency (n)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_visible(False)

    fig.subplots_adjust(top=0.98, bottom=0.18, left=0.22, right=0.93)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_reviewer_inputs()

    required_cols = [
        STAGE_COL,
        Q2_ABSTRACT_COL,
        Q2_TERMS_COL,
        AI_TASK_COL,
        TITLE_COL,
        ABSTRACT_COL,
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}\nAvailable columns: {list(df.columns)}"
        )

    df["stage_norm"] = df[STAGE_COL].apply(normalize_stage)
    df["q2_abstract_bool"] = df[Q2_ABSTRACT_COL].apply(normalize_bool_signal)
    df["q2_terms_bool"] = df[Q2_TERMS_COL].apply(normalize_bool_signal)

    df["title_norm"] = df[TITLE_COL].apply(norm_text)
    df["abstract_norm"] = df[ABSTRACT_COL].apply(norm_text)
    df["ai_task_norm"] = df[AI_TASK_COL].apply(norm_text)
    df["notes_norm"] = df[NOTES_COL].apply(norm_text) if NOTES_COL in df.columns else ""

    df["integration_approach"] = df.apply(derive_integration_approach, axis=1)
    df["integration_trace"] = df.apply(explain_integration_approach, axis=1)
    df["decision_timing"] = df.apply(derive_decision_timing, axis=1)
    df["decision_timing_trace"] = df.apply(explain_decision_timing, axis=1)
    df["review_trace"] = df.apply(build_review_reasons, axis=1)

    df.to_csv(outdir / "rq2_tripartite_dataset.csv", index=False)

    stage_approach = pd.crosstab(df["stage_norm"], df["integration_approach"])
    approach_timing = pd.crosstab(df["integration_approach"], df["decision_timing"])
    stage_approach.to_csv(outdir / "rq2_stage_x_approach.csv")
    approach_timing.to_csv(outdir / "rq2_approach_x_timing.csv")

    draw_tripartite_figure(df, outdir / "rq2_tripartite_integration_model.png")

    review_rows = df[
        df["integration_approach"].isin(["Unspecified integration", "No clear integration", "Not specified"])
        | df["decision_timing"].eq("Unspecified")
        | df["stage_norm"].eq("Not specified")
        | df["q2_abstract_bool"].isna()
        | df["q2_terms_bool"].isna()
    ].copy()

    preferred_cols = [
        STUDY_ID_COL,
        YEAR_COL,
        DOI_COL,
        MODALITY_COL,
        TITLE_COL,
        STAGE_COL,
        "stage_norm",
        Q2_ABSTRACT_COL,
        Q2_TERMS_COL,
        AI_TASK_COL,
        NOTES_COL,
        "integration_approach",
        "integration_trace",
        "decision_timing",
        "decision_timing_trace",
        "review_trace",
    ]
    preferred_cols = [c for c in preferred_cols if c in review_rows.columns]
    review_rows = review_rows[preferred_cols]
    review_rows.to_csv(outdir / "rq2_review_rows.csv", index=False)

    print(f"RQ2 outputs saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
