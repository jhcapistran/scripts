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
# Same corrected consolidated file used by RQ2/RQ3 (fixes stage_primary gaps present in the older file).
INPUT_FILE = BASE_DIR / "consolidado_RA_RB_Q3_completado_RQ2_final.xlsx"
SHEET_NAME = "Consolidado_por_asignacion"
OUTPUT_DIR = BASE_DIR / "rq1_results_q1_v2"
SUPPORTING_DIR = BASE_DIR / "rq1_supporting_q1_v2"
MASTER_DENOMINATOR_FILE = BASE_DIR / "rq_denominators_q1_v2.csv"

MODALITY_COL = "modalidad"
STAGE_COL = "stage_primary"
METHOD_COL = "tipo_IA"
ALGORITHM_COL = "AI_algorithm_main"
NOTES_COL = "notes_coding"

STAGE_FLAG_COLS = {
    "stage_prescreening": "Prescreening",
    "stage_screening": "Screening",
    "stage_diagnosis": "Diagnosis",
    "stage_prognosis": "Prognosis",
    "stage_monitoring_intervention": "Monitoring/intervention",
}
STAGE_ORDER = [
    "Prescreening",
    "Screening",
    "Diagnosis",
    "Prognosis",
    "Monitoring/intervention",
    "Not specified",
]
FIGURE_STAGE_ORDER = [stage for stage in STAGE_ORDER if stage != "Not specified"]
MODALITY_ORDER = [
    "Image",
    "Physiological signals",
    "Text / NLP",
    "Audio / Voice",
    "Multimodal",
    "Not specified",
]
METHOD_ORDER = ["Machine Learning", "Deep Learning", "Hybrid", "Not specified"]

MODALITY_TRANSLATIONS = {
    "imagen": "Image",
    "señales fisiológicas": "Physiological signals",
    "senales fisiologicas": "Physiological signals",
    "seã±ales fisiolã³gicas": "Physiological signals",
    "seã£â±ales fisiolã£â³gicas": "Physiological signals",
    "texto · nlp": "Text / NLP",
    "texto / nlp": "Text / NLP",
    "texto â· nlp": "Text / NLP",
    "audio · voz": "Audio / Voice",
    "audio / voz": "Audio / Voice",
    "audio â· voz": "Audio / Voice",
    "multimodal": "Multimodal",
    "no especificado": "Not specified",
    "no especificada": "Not specified",
    "not specified": "Not specified",
}
METHOD_TRANSLATIONS = {
    "machine learning": "Machine Learning",
    "deep learning": "Deep Learning",
    "híbrido": "Hybrid",
    "hibrido": "Hybrid",
    "hybrid": "Hybrid",
    "no especificado": "Not specified",
    "not specified": "Not specified",
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


def normalize_category(value: object, mapping: dict[str, str], fallback: str = "Not specified") -> str:
    cleaned = clean_text(value)
    if pd.isna(cleaned):
        return fallback
    return mapping.get(str(cleaned).casefold(), str(cleaned))


def normalize_stage(value: object) -> str:
    stage = normalize_category(value, STAGE_TRANSLATIONS)
    return stage if stage in STAGE_ORDER else "Not specified"


def normalize_algorithm(value: object) -> str:
    cleaned = clean_text(value)
    if pd.isna(cleaned):
        return "Not specified"
    text = str(cleaned)
    if text.casefold() in {"other / not clear", "other/not clear", "not clear", "other", "no especificado", "not specified"}:
        return "Not specified"
    return text


def normalize_bool_signal(value: object) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def ordered_categories(observed: list[str], preferred: list[str]) -> list[str]:
    ordered = [item for item in preferred if item in observed]
    extras = sorted(item for item in observed if item not in preferred)
    return ordered + extras


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


def load_base_df() -> pd.DataFrame:
    return pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME).copy()


def resolve_single_stage(row: pd.Series) -> tuple[str, str, int, str]:
    primary = normalize_stage(row.get(STAGE_COL))
    stages = [label for col, label in STAGE_FLAG_COLS.items() if normalize_bool_signal(row.get(col))]
    n_flags = len(stages)
    if primary in STAGE_ORDER and primary != "Not specified":
        if n_flags == 0:
            return primary, "stage_primary", 0, ""
        if primary in stages:
            return primary, "stage_primary_confirmed_by_flags", n_flags, ""
        return primary, "stage_primary_overrode_flags", n_flags, "|".join(stages)
    if n_flags == 1:
        return stages[0], "single_stage_flag_fallback", n_flags, ""
    if n_flags > 1:
        return "Not specified", "multiple_stage_flags_without_primary", n_flags, "|".join(stages)
    return "Not specified", "unresolved", 0, ""


def scale_bubble_sizes(counts: pd.Series, min_size: float = 850, max_size: float = 4200) -> pd.Series:
    counts = counts.astype(float)
    if counts.empty or counts.max() <= 0:
        return pd.Series(min_size, index=counts.index, dtype=float)
    normalized = np.sqrt(counts / counts.max())
    return min_size + normalized * (max_size - min_size)


def wrap_stage_label(label: str) -> str:
    return "Monitoring\nintervention" if label == "Monitoring/intervention" else label


def wrap_algorithm_label(label: str) -> str:
    mapping = {
        "Logistic Regression": "Logistic\nRegression",
        "Gradient boosting": "Gradient\nboosting",
    }
    return mapping.get(label, label)


def build_reference_algorithm_bubble(counts_df: pd.DataFrame, out_stem: Path) -> None:
    work = counts_df.groupby(["stage", "algorithm_display"], as_index=False)["count"].sum()
    work = work[work["count"] > 0].copy()
    if work.empty:
        return
    stage_order = [stage for stage in FIGURE_STAGE_ORDER if stage in work["stage"].unique()]
    algorithm_order = (
        work.groupby("algorithm_display", as_index=False)["count"]
        .sum()
        .sort_values(["count", "algorithm_display"], ascending=[True, True])["algorithm_display"]
        .tolist()
    )
    x_map = {stage: idx for idx, stage in enumerate(stage_order)}
    y_map = {algorithm: idx for idx, algorithm in enumerate(algorithm_order)}
    work["x_pos"] = work["stage"].map(x_map)
    work["y_pos"] = work["algorithm_display"].map(y_map)
    work["size"] = scale_bubble_sizes(work["count"])
    fig, ax = plt.subplots(figsize=(11.6, 10.4), facecolor="white")
    ax.scatter(work["x_pos"], work["y_pos"], s=work["size"], c=work["count"], cmap="viridis", alpha=0.82, edgecolors="none")
    for _, row in work.iterrows():
        ax.text(row["x_pos"], row["y_pos"], str(int(row["count"])), ha="center", va="center", fontsize=13, color="white", fontweight="bold")
    ax.set_xticks(range(len(stage_order)), [wrap_stage_label(stage) for stage in stage_order])
    ax.set_yticks(range(len(algorithm_order)), [wrap_algorithm_label(label) for label in algorithm_order])
    ax.set_xlabel("Clinical stage", fontsize=16)
    ax.set_ylabel("Algorithm family", fontsize=16)
    ax.tick_params(axis="both", labelsize=13, length=0)
    ax.grid(True, color="#e5e7eb", linewidth=1.0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    save_figure_variants(fig, out_stem)


def build_method_source_stage_heatmaps(counts_df: pd.DataFrame, out_stem: Path) -> None:
    work = counts_df[counts_df["count"] > 0].copy()
    if work.empty:
        return
    stage_order = [stage for stage in FIGURE_STAGE_ORDER if stage in work["stage"].unique()]
    modality_order = [item for item in MODALITY_ORDER if item in work["source_modality_display"].unique()]
    method_order = [item for item in METHOD_ORDER if item in work["ai_method_display"].unique()]
    matrix_blocks = []
    xtick_positions: list[float] = []
    xtick_labels: list[str] = []
    stage_boundaries: list[float] = []
    stage_centers: list[tuple[float, str]] = []
    current_col = 0
    for stage in stage_order:
        subset = work[work["stage"] == stage]
        table = (
            subset.pivot(index="source_modality_display", columns="ai_method_display", values="count")
            .reindex(index=modality_order, columns=method_order, fill_value=0)
            .fillna(0)
        )
        matrix_blocks.append(table.to_numpy())
        for offset, method in enumerate(method_order):
            xtick_positions.append(current_col + offset)
            xtick_labels.append(method)
        stage_centers.append((current_col + (len(method_order) - 1) / 2, stage))
        current_col += len(method_order)
        if stage != stage_order[-1]:
            stage_boundaries.append(current_col - 0.5)
    full_matrix = np.concatenate(matrix_blocks, axis=1)
    vmax = int(full_matrix.max()) if full_matrix.size else 0
    fig, ax = plt.subplots(figsize=(14.8, 7.0), facecolor="white")
    im = ax.imshow(full_matrix, cmap="YlGnBu", vmin=0, vmax=vmax, aspect="auto")
    for row_idx in range(full_matrix.shape[0]):
        for col_idx in range(full_matrix.shape[1]):
            value = int(full_matrix[row_idx, col_idx])
            color = "white" if value >= max(vmax * 0.45, 1) else "#16324f"
            ax.text(col_idx, row_idx, str(value), ha="center", va="center", fontsize=10.5, color=color, fontweight="bold" if value > 0 else None)
    ax.set_xticks(xtick_positions, xtick_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(modality_order)), modality_order)
    ax.tick_params(axis="both", labelsize=11.5, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for boundary in stage_boundaries:
        ax.axvline(boundary, color="#94a3b8", linewidth=1.7)
    for center, stage in stage_centers:
        ax.text(center, 1.06, wrap_stage_label(stage).replace("\n", " "), transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=13, color="#243b5a", fontweight="bold")
    ax.set_xlabel("AI method grouped by clinical stage", fontsize=16, labelpad=20)
    ax.set_ylabel("Data source", fontsize=16)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=11)
    fig.subplots_adjust(left=0.18, right=0.91, bottom=0.23, top=0.89)
    save_figure_variants(fig, out_stem)


def classify_note_tags(note: str) -> list[str]:
    rules = [
        ("Algorithm not explicit or unspecified", (r"no explícito", r"no explicito", r"no especificado", r"not specified", r"not explicit")),
        ("Needs full-text confirmation", (r"full[- ]text", r"confirmar")),
        ("Outside codebook or forced mapping", (r"fuera del codebook", r"outside (the )?codebook", r"\bmapped to\b", r"\bmapea\b")),
    ]
    tags: list[str] = []
    for label, patterns in rules:
        if any(re.search(pattern, note, flags=re.IGNORECASE) for pattern in patterns):
            tags.append(label)
    return tags


def run_notes_audit(df: pd.DataFrame, outdir: Path) -> None:
    if NOTES_COL not in df.columns:
        return
    notes_df = df.copy()
    notes_df[NOTES_COL] = notes_df[NOTES_COL].fillna("").astype(str).str.strip()
    notes_df = notes_df[notes_df[NOTES_COL].astype(bool)].copy()
    if notes_df.empty:
        return
    notes_df["notes_issue_tags"] = notes_df[NOTES_COL].map(lambda note: " | ".join(classify_note_tags(note)))
    notes_df.to_csv(outdir / "rq1_notes_full_context.csv", index=False)
    summary_lines = [
        "RQ1 notes support summary",
        f"Notes column: {NOTES_COL}",
        f"Rows with non-empty notes: {len(notes_df)}",
        f"Full note trail file: {outdir / 'rq1_notes_full_context.csv'}",
    ]
    (outdir / "rq1_notes_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def cleanup_outputs(directory: Path, prefix: str) -> None:
    keep = {
        "rq1_algorithm_bubbles_q1_v2.pdf",
        "rq1_algorithm_bubbles_q1_v2.png",
        "rq1_algorithm_bubbles_q1_v2.svg",
        "rq1_algorithm_plot_mapping.csv",
        "rq1_counts_algorithm_x_source_modality_by_stage.csv",
        "rq1_counts_source_modality_x_method_by_stage.csv",
        "rq1_method_source_stage_heatmap_q1_v2.pdf",
        "rq1_method_source_stage_heatmap_q1_v2.png",
        "rq1_method_source_stage_heatmap_q1_v2.svg",
        "rq1_method_source_stage_heatmap_q1_v2_caption.txt",
        "rq1_stage_assignment_audit.csv",
        "rq1_stage_assignment_summary.csv",
        "rq1_notes_full_context.csv",
        "rq1_notes_summary.txt",
        "rq1_denominators_q1_v2.csv",
    }
    for path in directory.glob(f"{prefix}*"):
        if path.is_file() and path.name not in keep:
            path.unlink()


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    ensure_dir(SUPPORTING_DIR)
    df = load_base_df()
    df["source_modality_display"] = df[MODALITY_COL].apply(normalize_category, mapping=MODALITY_TRANSLATIONS)
    df["ai_method_display"] = df[METHOD_COL].apply(normalize_category, mapping=METHOD_TRANSLATIONS)
    df["algorithm_display_raw"] = df[ALGORITHM_COL].apply(normalize_algorithm)
    single_stage = df.apply(resolve_single_stage, axis=1, result_type="expand")
    df["resolved_stage_single"] = single_stage[0]
    df["resolved_stage_origin_single"] = single_stage[1]
    df["resolved_stage_flag_count"] = single_stage[2]
    df["resolved_stage_flag_labels"] = single_stage[3]
    analyzed = df[df["resolved_stage_single"].notna()].copy()
    analyzed["stage"] = analyzed["resolved_stage_single"]
    algorithm_counts = (
        analyzed.groupby(["stage", "algorithm_display_raw", "source_modality_display"])["study_id"]
        .nunique()
        .reset_index(name="count")
        .rename(columns={"algorithm_display_raw": "algorithm_display"})
    )
    method_counts = (
        analyzed.groupby(["stage", "source_modality_display", "ai_method_display"])["study_id"]
        .nunique()
        .reset_index(name="count")
    )
    algorithm_counts.to_csv(OUTPUT_DIR / "rq1_counts_algorithm_x_source_modality_by_stage.csv", index=False)
    method_counts.to_csv(OUTPUT_DIR / "rq1_counts_source_modality_x_method_by_stage.csv", index=False)
    build_reference_algorithm_bubble(algorithm_counts, OUTPUT_DIR / "rq1_algorithm_bubbles_q1_v2")
    build_method_source_stage_heatmaps(method_counts, OUTPUT_DIR / "rq1_method_source_stage_heatmap_q1_v2")
    write_caption(
        OUTPUT_DIR / "rq1_method_source_stage_heatmap_q1_v2_caption.txt",
        """
        RQ1 companion figure. Faceted heatmaps showing the distribution of AI methods across data sources within
        each functional stage of the clinical process in ASD. Each panel corresponds to one clinical stage; rows
        represent data sources, columns represent AI methods, and cell labels report counts directly.
        """,
    )
    analyzed[[
        "study_id",
        "title",
        "year",
        "doi",
        "source_modality_display",
        "ai_method_display",
        "algorithm_display_raw",
        STAGE_COL,
        "resolved_stage_single",
        "resolved_stage_origin_single",
        "resolved_stage_flag_count",
        "resolved_stage_flag_labels",
    ]].to_csv(OUTPUT_DIR / "rq1_stage_assignment_audit.csv", index=False)
    (
        analyzed["resolved_stage_origin_single"]
        .value_counts()
        .rename_axis("resolved_stage_origin_single")
        .reset_index(name="studies_n")
        .to_csv(OUTPUT_DIR / "rq1_stage_assignment_summary.csv", index=False)
    )
    analyzed[["algorithm_display_raw"]].drop_duplicates().sort_values("algorithm_display_raw").rename(columns={"algorithm_display_raw": "algorithm_raw"}).assign(
        algorithm_plot=lambda x: x["algorithm_raw"]
    ).to_csv(OUTPUT_DIR / "rq1_algorithm_plot_mapping.csv", index=False)
    run_notes_audit(analyzed, SUPPORTING_DIR)
    stage_resolved_n = int((analyzed["resolved_stage_single"] != "Not specified").sum())
    denominator_rows = [
        {"rq": "RQ1", "subset": "all_rows", "group": "Evaluated studies", "denominator_n": int(len(df)), "notes": "Analytical universe: graph-ready studies evaluated for RQ1."},
        {"rq": "RQ1", "subset": "stage_resolution", "group": "Stage resolved", "denominator_n": stage_resolved_n, "notes": "resolved_stage_single not equal to 'Not specified'."},
        {"rq": "RQ1", "subset": "stage_resolution", "group": "Not specified", "denominator_n": int(len(df)) - stage_resolved_n, "notes": "resolved_stage_single equal to 'Not specified'."},
    ]
    pd.DataFrame(denominator_rows).to_csv(OUTPUT_DIR / "rq1_denominators_q1_v2.csv", index=False)
    cleanup_outputs(OUTPUT_DIR, "rq1_")
    cleanup_outputs(SUPPORTING_DIR, "rq1_")


def _self_check() -> None:
    assert normalize_stage("screening") == "Screening"


if __name__ == "__main__":
    _self_check()
    main()
