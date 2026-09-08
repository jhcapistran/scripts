from __future__ import annotations

from pathlib import Path
import re

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd
import scienceplots


matplotlib.use("Agg")
plt.style.use(["science", "no-latex"])

BASE_DIR = Path(__file__).resolve().parent
MASTER_FILE = BASE_DIR / "Maestro_IA_TEA_cierre_2026-09-08.xlsx"
SHEET_NAME = "BASE_CIERRE"
OUTPUT_DIR = BASE_DIR / "rq1_results_q1_v2"
SUPPORTING_DIR = BASE_DIR / "rq1_supporting_q1_v2"

STAGE_ORDER = [
    "Prescreening",
    "Screening",
    "Diagnosis",
    "Prognosis",
    "Monitoring/intervention",
    "Not specified",
]
FIGURE_STAGE_ORDER = STAGE_ORDER

MODALITY_ORDER = [
    "Neuroimaging",
    "Structured clinical/questionnaire data",
    "Behavioral images/video/gaze/movement",
    "Multiple source categories",
    "Physiological signals",
    "Biological/omics",
    "Text/language",
    "Audio/voice",
    "Other biomedical/digital data",
    "Not specified",
]

METHOD_ORDER = ["Machine Learning", "Deep Learning", "Hybrid", "Not specified"]


def clean_text(value: object) -> str:
    if pd.isna(value):
        return "Not specified"
    text = str(value).strip()
    return text if text else "Not specified"


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
    if not MASTER_FILE.exists():
        raise FileNotFoundError(f"Master file not found: {MASTER_FILE}")
    df = pd.read_excel(MASTER_FILE, sheet_name=SHEET_NAME, skiprows=3)
    df = df[df["include_main"] == 1].copy()
    if len(df) != 428:
        raise ValueError(f"Expected 428 included studies in BASE_CIERRE, found {len(df)}")
    return df


def scale_bubble_sizes(counts: pd.Series, min_size: float = 850, max_size: float = 4200) -> pd.Series:
    counts = counts.astype(float)
    if counts.empty or counts.max() <= 0:
        return pd.Series(min_size, index=counts.index, dtype=float)
    normalized = np.sqrt(counts / counts.max())
    return min_size + normalized * (max_size - min_size)


def wrap_stage_label(label: str) -> str:
    return "Monitoring/\nintervention" if label == "Monitoring/intervention" else label


def wrap_algorithm_label(label: str) -> str:
    mapping = {
        "Composite/ensemble pipeline": "Composite/ensemble\npipeline",
        "Multiple/not uniquely specified": "Multiple/not\nuniquely specified",
        "Other specified method": "Other specified\nmethod",
        "Decision tree/rules": "Decision tree/\nrules",
        "Logistic regression": "Logistic\nregression",
        "Gradient boosting": "Gradient\nboosting",
        "k-nearest neighbors": "k-nearest\nneighbors",
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
    work["size"] = scale_bubble_sizes(work["count"], min_size=800, max_size=3800)
    fig, ax = plt.subplots(figsize=(12.2, 11.0), facecolor="white")
    ax.scatter(work["x_pos"], work["y_pos"], s=work["size"], c=work["count"], cmap="viridis", alpha=0.85, edgecolors="none")
    for _, row in work.iterrows():
        ax.text(row["x_pos"], row["y_pos"], str(int(row["count"])), ha="center", va="center", fontsize=12.5, color="white", fontweight="bold")
    ax.set_xticks(range(len(stage_order)), [wrap_stage_label(stage) for stage in stage_order])
    ax.set_yticks(range(len(algorithm_order)), [wrap_algorithm_label(label) for label in algorithm_order])
    ax.set_xlabel("Clinical stage", fontsize=16, labelpad=12)
    ax.set_ylabel("Algorithm family", fontsize=16, labelpad=12)
    ax.tick_params(axis="both", labelsize=12.5, length=0)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
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
    fig, ax = plt.subplots(figsize=(16.0, 7.8), facecolor="white")
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
    fig.subplots_adjust(left=0.22, right=0.92, bottom=0.24, top=0.88)
    save_figure_variants(fig, out_stem)


def run_notes_audit(df: pd.DataFrame, outdir: Path) -> None:
    notes_rows = []
    for _, row in df.iterrows():
        notes = []
        if pd.notna(row.get("rationale")):
            notes.append(f"Rationale: {row['rationale']}")
        if pd.notna(row.get("source_scope_caution_detail")):
            notes.append(f"Scope caution: {row['source_scope_caution_detail']}")
        if pd.notna(row.get("integrity_caution")):
            notes.append(f"Integrity caution: {row['integrity_caution']}")
        if notes:
            notes_rows.append({
                "study_id": row["study_id"],
                "cohort": row.get("cohort", ""),
                "title": row["title"],
                "notes_combined": " | ".join(notes),
            })
    notes_df = pd.DataFrame(notes_rows)
    if not notes_df.empty:
        notes_df.to_csv(outdir / "rq1_notes_full_context.csv", index=False)
    summary_lines = [
        "RQ1 notes support summary",
        f"Master file: {MASTER_FILE.name}",
        f"Included studies with audit notes: {len(notes_df)}",
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

    df["stage"] = df["stage_primary"].astype(str)
    df["source_modality_display"] = df["data_source_primary"].astype(str)
    df["ai_method_display"] = df["ai_type"].astype(str)
    df["algorithm_display"] = df["algorithm_family"].astype(str)
    df["algorithm_main_clean"] = df["algorithm_main"].astype(str)

    algorithm_counts = (
        df.groupby(["stage", "algorithm_display", "source_modality_display"])["study_id"]
        .nunique()
        .reset_index(name="count")
    )
    method_counts = (
        df.groupby(["stage", "source_modality_display", "ai_method_display"])["study_id"]
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
        represent data sources, columns represent AI methods, and cell labels report counts directly from the final
        adjudicated master workbook (N=428 included reports).
        """,
    )

    df[[
        "study_id",
        "cohort",
        "title",
        "year",
        "doi",
        "source_modality_display",
        "ai_method_display",
        "algorithm_main_clean",
        "algorithm_display",
        "stage",
    ]].rename(columns={
        "source_modality_display": "data_source_primary",
        "ai_method_display": "ai_type",
        "algorithm_main_clean": "algorithm_main",
        "algorithm_display": "algorithm_family",
        "stage": "stage_primary",
    }).to_csv(OUTPUT_DIR / "rq1_stage_assignment_audit.csv", index=False)

    (
        df["stage"]
        .value_counts()
        .rename_axis("stage_primary")
        .reset_index(name="studies_n")
        .to_csv(OUTPUT_DIR / "rq1_stage_assignment_summary.csv", index=False)
    )

    mapping_df = (
        df[["algorithm_main_clean", "algorithm_display"]]
        .drop_duplicates()
        .sort_values(["algorithm_display", "algorithm_main_clean"])
        .rename(columns={
            "algorithm_main_clean": "algorithm_raw",
            "algorithm_display": "algorithm_plot",
        })
    )
    mapping_df.to_csv(OUTPUT_DIR / "rq1_algorithm_plot_mapping.csv", index=False)

    run_notes_audit(df, SUPPORTING_DIR)

    stage_resolved_n = int((df["stage"] != "Not specified").sum())
    denominator_rows = [
        {"rq": "RQ1", "subset": "all_rows", "group": "Evaluated studies", "denominator_n": int(len(df)), "notes": "Analytical universe: included reports evaluated for RQ1 from Maestro_IA_TEA_cierre_2026-09-08.xlsx."},
        {"rq": "RQ1", "subset": "stage_resolution", "group": "Stage resolved", "denominator_n": stage_resolved_n, "notes": "stage_primary resolved to one of the five clinical functional stages."},
        {"rq": "RQ1", "subset": "stage_resolution", "group": "Not specified", "denominator_n": int(len(df)) - stage_resolved_n, "notes": "stage_primary not uniquely specified in assessed material."},
    ]
    pd.DataFrame(denominator_rows).to_csv(OUTPUT_DIR / "rq1_denominators_q1_v2.csv", index=False)

    cleanup_outputs(OUTPUT_DIR, "rq1_")
    cleanup_outputs(SUPPORTING_DIR, "rq1_")


def _self_check() -> None:
    assert len(STAGE_ORDER) == 6
    assert len(METHOD_ORDER) == 4


if __name__ == "__main__":
    _self_check()
    main()
