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
import pandas as pd


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
    timing_order = ["Pre-decision", "In-decision", "Post-decision", "Unspecified"]

    stage_labels = [x for x in stage_order if x in set(df["stage_norm"])]
    approach_labels = [x for x in approach_order if x in set(df["integration_approach"])]
    timing_labels = [x for x in timing_order if x in set(df["decision_timing"])]

    x_positions = {"stage": 0.12, "approach": 0.5, "timing": 0.88}
    y_stage = compute_node_positions(stage_labels)
    y_approach = compute_node_positions(approach_labels)
    y_timing = compute_node_positions(timing_labels)

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    stage_to_approach = (
        df.groupby(["stage_norm", "integration_approach"])
        .size()
        .reset_index(name="count")
    )
    approach_to_timing = (
        df.groupby(["integration_approach", "decision_timing"])
        .size()
        .reset_index(name="count")
    )

    max_edge = max(
        stage_to_approach["count"].max() if not stage_to_approach.empty else 1,
        approach_to_timing["count"].max() if not approach_to_timing.empty else 1,
    )

    for row in stage_to_approach.itertuples(index=False):
        ax.plot(
            [x_positions["stage"], x_positions["approach"]],
            [y_stage[row.stage_norm], y_approach[row.integration_approach]],
            color="#7aa6c2",
            linewidth=1.5 + 8 * (row.count / max_edge),
            alpha=0.45,
            solid_capstyle="round",
            zorder=1,
        )

    for row in approach_to_timing.itertuples(index=False):
        ax.plot(
            [x_positions["approach"], x_positions["timing"]],
            [y_approach[row.integration_approach], y_timing[row.decision_timing]],
            color="#d28f5a",
            linewidth=1.5 + 8 * (row.count / max_edge),
            alpha=0.45,
            solid_capstyle="round",
            zorder=1,
        )

    stage_sizes = df["stage_norm"].value_counts()
    approach_sizes = df["integration_approach"].value_counts()
    timing_sizes = df["decision_timing"].value_counts()
    max_node = max(stage_sizes.max(), approach_sizes.max(), timing_sizes.max())

    def draw_layer(labels: list[str], x: float, y_map: dict[str, float], counts: pd.Series, color: str) -> None:
        for label in labels:
            size = 1200 + 4200 * (counts[label] / max_node)
            ax.scatter(
                [x],
                [y_map[label]],
                s=size,
                color=color,
                alpha=0.95,
                zorder=3,
                edgecolors="white",
                linewidths=1.8,
            )
            offset = -0.035 if x > 0.5 else 0.035
            align = "right" if x > 0.5 else "left"
            if x == x_positions["approach"]:
                offset = 0
                align = "center"
            ax.text(
                x + offset,
                y_map[label],
                f"{wrap_label(label)}\n(n={int(counts[label])})",
                ha=align,
                va="center",
                fontsize=10,
                zorder=4,
            )

    draw_layer(stage_labels, x_positions["stage"], y_stage, stage_sizes, "#355070")
    draw_layer(approach_labels, x_positions["approach"], y_approach, approach_sizes, "#6d597a")
    draw_layer(timing_labels, x_positions["timing"], y_timing, timing_sizes, "#b56576")

    ax.text(x_positions["stage"], 0.98, "Clinical stage", ha="center", va="top", fontsize=13, fontweight="bold")
    ax.text(x_positions["approach"], 0.98, "Integration approach", ha="center", va="top", fontsize=13, fontweight="bold")
    ax.text(x_positions["timing"], 0.98, "Decision timing", ha="center", va="top", fontsize=13, fontweight="bold")

    fig.tight_layout()
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
