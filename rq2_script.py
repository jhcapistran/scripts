from __future__ import annotations

from pathlib import Path
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots


matplotlib.use("Agg")
plt.style.use(["science", "no-latex"])

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "consolidado_RA_RB_Q3_completado_RQ2_final.xlsx"
SHEET_NAME = "Consolidado_por_asignacion"
OUTPUT_DIR = BASE_DIR / "rq2_results_q1_v2"
# Confirmed non-primary records (reviews/surveys/perspectives) shared across RQ1/RQ2/RQ3, see rq_eligibility_recheck_q1_v2.csv.
EXCLUDED_STUDIES_FILE = BASE_DIR / "rq_excluded_studies_q1_v2.csv"

STAGE_ORDER = [
    "Prescreening",
    "Screening",
    "Diagnosis",
    "Prognosis",
    "Monitoring/intervention",
    "Not specified",
]

INTEGRATION_LABELS = {
    "Triage / questionnaires": "Triage /\nquestionnaires",
    "Mobile screening": "Mobile\nscreening",
    "Feature extraction": "Feature\nextraction",
    "Second-reader decision support": "Second-reader\ndecision support",
    "Risk stratification": "Risk\nstratification",
    "Longitudinal dashboards": "Longitudinal\ndashboards",
    "Adaptive intervention": "Adaptive\nintervention",
    "Assistive tools": "Assistive\ntools",
}
STAGE_TRANSLATIONS = {
    "prescreening": "Prescreening",
    "screening": "Screening",
    "diagnosis": "Diagnosis",
    "prognosis": "Prognosis",
    "monitoring/intervention": "Monitoring/intervention",
    "monitoring_intervention": "Monitoring/intervention",
    "not clear": "Not specified",
    "not specified": "Not specified",
    "no especificado": "Not specified",
}


def clean_text(value: object) -> object:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    return text if text else pd.NA


def norm_text(value: object) -> str:
    cleaned = clean_text(value)
    if pd.isna(cleaned):
        return ""
    return re.sub(r"\s+", " ", str(cleaned).casefold().replace("\n", " "))


def normalize_category(value: object, mapping: dict[str, str], fallback: str = "Not specified") -> str:
    cleaned = clean_text(value)
    if pd.isna(cleaned):
        return fallback
    return mapping.get(str(cleaned).casefold(), str(cleaned))


def normalize_stage(value: object) -> str:
    stage = normalize_category(value, STAGE_TRANSLATIONS)
    return stage if stage in STAGE_ORDER else "Not specified"


def normalize_bool_signal(value: object) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        if float(value) > 0:
            return True
        if float(value) == 0:
            return False
    text = norm_text(value)
    if text in {"verdadero", "true", "1", "1.0", "yes", "y"}:
        return True
    if text in {"falso", "false", "0", "0.0", "no", "n"}:
        return False
    return None


def status_from_bool(value: bool | None) -> str:
    if value is True:
        return "Present"
    if value is False:
        return "Absent"
    return "Uncoded"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_figure_variants(fig: plt.Figure, stem: Path) -> None:
    for suffix in (".png", ".pdf", ".svg"):
        outpath = stem.with_suffix(suffix)
        kwargs = {"bbox_inches": "tight"}
        if suffix == ".png":
            kwargs["dpi"] = 600
        fig.savefig(outpath, **kwargs)
    plt.close(fig)


def write_caption(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def wrap_integration_label(label: str) -> str:
    return INTEGRATION_LABELS.get(label, label)


def ordered_categories(observed: list[str], preferred: list[str]) -> list[str]:
    ordered = [item for item in preferred if item in observed]
    extras = sorted(item for item in observed if item not in preferred)
    return ordered + extras


def load_excluded_study_ids() -> pd.DataFrame:
    if not EXCLUDED_STUDIES_FILE.exists():
        return pd.DataFrame(columns=["study_id", "tier", "title", "reason"])
    return pd.read_csv(EXCLUDED_STUDIES_FILE)


def load_base_df() -> pd.DataFrame:
    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME).copy()
    excluded = load_excluded_study_ids()
    df = df[~df["study_id"].isin(excluded["study_id"])].copy()
    df["row_id"] = range(1, len(df) + 1)
    return df


def normalize_common_fields(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["stage_norm"] = work["stage_primary"].apply(normalize_stage)
    return work


def derive_q2_signal_state(row: pd.Series) -> str:
    abs_signal = normalize_bool_signal(row.get("q2_candidate_abstract"))
    terms_signal = normalize_bool_signal(row.get("q2_candidate_terms"))
    if abs_signal is True or terms_signal is True:
        return "Present"
    if abs_signal is False or terms_signal is False:
        return "Absent"
    return "Uncoded"


def derive_integration_approach_positive(row: pd.Series) -> str:
    stage = row["stage_norm"]
    text = " ".join(
        [
            norm_text(row.get("title")),
            norm_text(row.get("abstract")),
            norm_text(row.get("AI_task_type")),
            norm_text(row.get("notes_coding")),
        ]
    )
    if any(token in text for token in ["questionnaire", "checklist", "triage", "refer", "prescreen", "pre-screen"]):
        return "Triage / questionnaires"
    if any(token in text for token in ["mobile", "smartphone", "app-based", "tablet-based", "mhealth", "m-health"]):
        return "Mobile screening"
    if any(token in text for token in ["feature extraction", "feature selection", "biomarker extraction", "marker extraction"]):
        return "Feature extraction"
    if any(token in text for token in ["decision support", "computer-aided", "computer aided", "second reader", "second-reader", "clinician support"]):
        return "Second-reader decision support"
    if stage == "Prognosis" or any(token in text for token in ["risk stratification", "predictor", "prediction model", "prenatal", "perinatal", "maternal"]):
        return "Risk stratification"
    if any(token in text for token in ["dashboard", "longitudinal", "follow-up", "trajectory", "progress tracking"]):
        return "Longitudinal dashboards"
    if any(token in text for token in ["intervention", "therapy", "adaptive", "personalized", "virtual reality", "serious game", "robot-assisted"]):
        return "Adaptive intervention"
    if any(token in text for token in ["assistive", "educational", "school", "communication aid", "caregiver support"]):
        return "Assistive tools"
    if stage in {"Prescreening", "Screening"}:
        return "Triage / questionnaires"
    if stage == "Diagnosis":
        return "Second-reader decision support"
    if stage == "Monitoring/intervention":
        return "Adaptive intervention"
    return "Unspecified integration"


def derive_decision_timing(row: pd.Series) -> str:
    if row["q2_signal_state"] != "Present":
        return "Not applicable"
    approach = row["integration_approach"]
    text = " ".join(
        [
            norm_text(row.get("title")),
            norm_text(row.get("abstract")),
            norm_text(row.get("AI_task_type")),
            norm_text(row.get("notes_coding")),
        ]
    )
    if approach in {"Triage / questionnaires", "Mobile screening", "Feature extraction", "Risk stratification"}:
        return "Pre-decision"
    if approach == "Second-reader decision support":
        return "Pre-decision" if any(token in text for token in ["triage", "pre-read", "feature extraction"]) else "In-decision"
    if approach in {"Longitudinal dashboards", "Adaptive intervention", "Assistive tools"}:
        return "Post-decision"
    return "Unspecified"



def main() -> None:
    ensure_dir(OUTPUT_DIR)
    stale_timing_table = OUTPUT_DIR / "rq2_timing_table_q1_v2.csv"
    if stale_timing_table.exists():
        stale_timing_table.unlink()
    df = normalize_common_fields(load_base_df())
    df["q2_abstract_bool"] = df["q2_candidate_abstract"].map(normalize_bool_signal)
    df["q2_terms_bool"] = df["q2_candidate_terms"].map(normalize_bool_signal)
    df["q2_signal_state"] = df.apply(derive_q2_signal_state, axis=1)
    df["integration_approach"] = pd.NA
    positive_mask = df["q2_signal_state"].eq("Present")
    df.loc[positive_mask, "integration_approach"] = df.loc[positive_mask].apply(derive_integration_approach_positive, axis=1)
    df["decision_timing"] = df.apply(derive_decision_timing, axis=1)
    df["q2_signal_pattern"] = df.apply(
        lambda row: f"abstract={status_from_bool(row['q2_abstract_bool'])}; terms={status_from_bool(row['q2_terms_bool'])}",
        axis=1,
    )
    positive_df = df.loc[positive_mask].copy()
    integration_order = [
        "Triage / questionnaires",
        "Mobile screening",
        "Feature extraction",
        "Second-reader decision support",
        "Risk stratification",
        "Longitudinal dashboards",
        "Adaptive intervention",
        "Assistive tools",
        "Unspecified integration",
    ]
    stage_integration = (
        pd.crosstab(positive_df["stage_norm"], positive_df["integration_approach"])
        .reindex(index=ordered_categories(df["stage_norm"].unique().tolist(), STAGE_ORDER), columns=integration_order, fill_value=0)
    )
    stage_integration = stage_integration.loc[(stage_integration.sum(axis=1) > 0), (stage_integration.sum(axis=0) > 0)]
    stage_integration.to_csv(OUTPUT_DIR / "rq2_stage_x_integration_q1_v2.csv")
    df.to_csv(OUTPUT_DIR / "rq2_tripartite_dataset_q1_v2.csv", index=False)
    review_rows = positive_df[positive_df["integration_approach"].eq("Unspecified integration") | positive_df["decision_timing"].eq("Unspecified")].copy()
    review_rows.to_csv(OUTPUT_DIR / "rq2_review_rows_q1_v2.csv", index=False)
    denominator_rows = [
        {"rq": "RQ2", "subset": "all_rows", "group": "All studies", "denominator_n": int(len(df)), "notes": "Unique studies in the consolidated sheet, after excluding confirmed non-primary records."},
        {"rq": "RQ2", "subset": "exclusions_applied", "group": "Tier1 non-primary (reviews/surveys/perspectives)", "denominator_n": len(load_excluded_study_ids()), "notes": "Excluded before analysis; see rq_excluded_studies_q1_v2.csv."},
    ]
    for state, count in df["q2_signal_state"].value_counts(dropna=False).reindex(["Present", "Absent", "Uncoded"], fill_value=0).items():
        denominator_rows.append(
            {
                "rq": "RQ2",
                "subset": "integration_signal_state",
                "group": state,
                "denominator_n": int(count),
                "notes": "Derived from q2_candidate_abstract and q2_candidate_terms.",
            }
        )
    pd.DataFrame(denominator_rows).to_csv(OUTPUT_DIR / "rq2_denominators_q1_v2.csv", index=False)
    fig, heat_ax = plt.subplots(figsize=(14.2, 7.4), facecolor="white")
    heat_values = stage_integration.to_numpy(dtype=float)
    img = heat_ax.imshow(heat_values, cmap="Blues", aspect="auto")
    # Determine threshold for switching annotation colour (white on dark, black on light)
    vmin, vmax = heat_values.min(), heat_values.max()
    threshold = vmin + (vmax - vmin) * 0.55
    row_totals = heat_values.sum(axis=1)
    for row_idx in range(heat_values.shape[0]):
        for col_idx in range(heat_values.shape[1]):
            value = int(heat_values[row_idx, col_idx])
            share = 0.0 if row_totals[row_idx] == 0 else value / row_totals[row_idx]
            if value == 0:
                label = "0"
                font_color = "#888888"
            else:
                label = f"{value}\n{share:.0%}"
                font_color = "white" if heat_values[row_idx, col_idx] >= threshold else "#1a1a2e"
            heat_ax.text(
                col_idx, row_idx, label,
                ha="center", va="center",
                fontsize=10.4, fontweight="bold",
                color=font_color,
                linespacing=1.4,
            )
    heat_ax.set_xticks(range(len(stage_integration.columns)))
    heat_ax.set_xticklabels([wrap_integration_label(col) for col in stage_integration.columns], rotation=32, ha="right", fontsize=12)
    heat_ax.set_yticks(range(len(stage_integration.index)))
    heat_ax.set_yticklabels(stage_integration.index, fontsize=11.5)
    heat_ax.set_xlabel("Integration approach", fontsize=14.5, fontweight="bold", labelpad=14)
    heat_ax.set_ylabel("Clinical stage", fontsize=14, fontweight="bold", labelpad=12)
    # Add a colourbar for scale reference
    cbar = fig.colorbar(img, ax=heat_ax, shrink=0.7, pad=0.02)
    cbar.set_label("Study count", fontsize=11.5)
    cbar.ax.tick_params(labelsize=10.5)
    fig.tight_layout()
    save_figure_variants(fig, OUTPUT_DIR / "rq2_heatmap_and_timing_q1_v2")
    write_caption(
        OUTPUT_DIR / "rq2_heatmap_and_timing_q1_v2_caption.txt",
        """
        RQ2. Stage-by-integration heatmap. The heatmap shows study counts by primary clinical stage and derived
        integration approach among studies with an explicit integration signal. Workflow-oriented categories are assigned
        using keyword and stage fallback rules. Cell annotations report the count and the within-stage percentage. Study-level
        classifications are stored in rq2_tripartite_dataset_q1_v2.csv and denominators are listed in
        rq2_denominators_q1_v2.csv.
        """,
    )


def _self_check() -> None:
    assert derive_q2_signal_state(pd.Series({"q2_candidate_abstract": "true", "q2_candidate_terms": pd.NA})) == "Present"


if __name__ == "__main__":
    _self_check()
    main()
