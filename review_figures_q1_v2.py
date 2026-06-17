from __future__ import annotations

import math
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import pandas as pd
import scienceplots


plt.style.use(["science", "no-latex"])


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "consolidado_RA_RB_Q3_completado.xlsx"
SHEET_NAME = "Consolidado_por_asignacion"

RQ1_OUTPUT_DIR = BASE_DIR / "rq1_results_q1_v2"
RQ1_SUPPORTING_DIR = BASE_DIR / "rq1_supporting_q1_v2"
RQ2_OUTPUT_DIR = BASE_DIR / "rq2_results_q1_v2"
RQ3_OUTPUT_DIR = BASE_DIR / "rq3_results_q1_v2"
RQ3_SUPPLEMENT_DIR = RQ3_OUTPUT_DIR / "supplement"
MASTER_DENOMINATOR_FILE = BASE_DIR / "rq_denominators_q1_v2.csv"

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
METHOD_ORDER = [
    "Machine Learning",
    "Deep Learning",
    "Hybrid",
    "Not specified",
]
MODALITY_ORDER = [
    "Image",
    "Physiological signals",
    "Text / NLP",
    "Audio / Voice",
    "Multimodal",
    "Not specified",
]
METHOD_COLORS = {
    "Machine Learning": "#1f77b4",
    "Deep Learning": "#d62728",
    "Hybrid": "#2ca02c",
    "Not specified": "#7f7f7f",
}
MODALITY_COLORS = {
    "Image": "#4c78a8",
    "Physiological signals": "#72b7b2",
    "Text / NLP": "#54a24b",
    "Audio / Voice": "#eeca3b",
    "Multimodal": "#e45756",
    "Not specified": "#9d9da0",
}
STATUS_COLORS = {
    "Positive": "#2ca02c",
    "Negative": "#d62728",
    "Missing": "#9d9da0",
}
PRACTICE_COLS = {
    "q3_external_validation_signal": "External validation",
    "q3_multisource_strategy_signal": "Multisource integration",
    "q3_explainability_signal": "Model explainability",
    "q3_multisite_signal": "Cross-site robustness",
}
PRACTICE_ORDER = list(PRACTICE_COLS.keys())

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


def norm_text(value: object) -> str:
    cleaned = clean_text(value)
    if pd.isna(cleaned):
        return ""
    text = str(cleaned).casefold()
    text = text.replace("\n", " ")
    return re.sub(r"\s+", " ", text)


def normalize_category(
    value: object,
    mapping: dict[str, str],
    fallback: str = "Not specified",
) -> str:
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
        return "Positive"
    if value is False:
        return "Negative"
    return "Missing"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_figure_variants(fig: plt.Figure, stem: Path) -> list[Path]:
    generated: list[Path] = []
    for suffix in (".png", ".pdf", ".svg"):
        outpath = stem.with_suffix(suffix)
        kwargs = {"bbox_inches": "tight"}
        if suffix == ".png":
            kwargs["dpi"] = 600
        fig.savefig(outpath, **kwargs)
        generated.append(outpath)
    plt.close(fig)
    return generated


def write_caption(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def ordered_categories(observed: list[str], preferred: list[str]) -> list[str]:
    ordered = [item for item in preferred if item in observed]
    extras = sorted(item for item in observed if item not in preferred)
    return ordered + extras


def load_base_df() -> pd.DataFrame:
    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME).copy()
    df["row_id"] = range(1, len(df) + 1)
    return df


def expand_stage_assignments(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for _, row in df.iterrows():
        stages = []
        for col, label in STAGE_FLAG_COLS.items():
            signal = normalize_bool_signal(row.get(col))
            if signal is True:
                stages.append(label)
        stage_source = "stage_flags"
        if not stages:
            primary = normalize_stage(row.get("stage_primary"))
            stages = [primary]
            stage_source = "stage_primary_fallback"
        for stage in stages:
            record = row.to_dict()
            record["stage_resolved"] = stage
            record["stage_source"] = stage_source
            records.append(record)
    return pd.DataFrame.from_records(records)


def normalize_common_fields(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["stage_norm"] = work["stage_primary"].apply(normalize_stage)
    work["method_norm"] = work["tipo_IA"].apply(normalize_category, mapping=METHOD_TRANSLATIONS)
    work["modality_norm"] = work["modalidad"].apply(normalize_category, mapping=MODALITY_TRANSLATIONS)
    work["algorithm_norm"] = work["AI_algorithm_main"].apply(clean_text).fillna("Not specified")
    work["algorithm_norm"] = work["algorithm_norm"].replace(
        {
            "Other / Not clear": "Not specified",
            "Not clear": "Not specified",
            "Other": "Other algorithms",
        }
    )
    return work


def compress_algorithms(series: pd.Series, top_k: int = 8) -> tuple[pd.Series, list[str]]:
    specific = series[~series.isin(["Not specified", "Other algorithms"])]
    top = specific.value_counts().head(top_k).index.tolist()
    compressed = series.map(
        lambda value: value if value in top or value in {"Not specified", "Other algorithms"} else "Other algorithms"
    )
    order = top.copy()
    if (compressed == "Other algorithms").any():
        order.append("Other algorithms")
    if (compressed == "Not specified").any():
        order.append("Not specified")
    return compressed, order


def scale_bubble_sizes(
    counts: pd.Series,
    min_size: float = 140,
    max_size: float = 1900,
) -> pd.Series:
    if counts.empty:
        return counts.astype(float)
    counts = counts.astype(float)
    max_count = counts.max()
    if max_count <= 0:
        return pd.Series(min_size, index=counts.index, dtype=float)
    normalized = np.sqrt(counts / max_count)
    return min_size + normalized * (max_size - min_size)


def wrap_stage_label(label: str) -> str:
    if label == "Monitoring/intervention":
        return "Monitoring\nintervention"
    return label


def wrap_algorithm_label(label: str) -> str:
    mapping = {
        "Logistic Regression": "Logistic\nRegression",
        "Gradient boosting": "Gradient\nboosting",
        "Other / unclear": "Other / unclear",
        "RNN/LSTM/GRU": "RNN/LSTM/GRU",
    }
    return mapping.get(label, label)


def build_reference_algorithm_bubble(
    counts_df: pd.DataFrame,
    out_stem: Path,
) -> None:
    work = counts_df.copy()
    work = work.groupby(["stage", "algorithm_display"], as_index=False)["count"].sum()
    work = work[work["count"] > 0].copy()
    if work.empty:
        return

    stage_order = [stage for stage in STAGE_ORDER if stage in work["stage"].unique()]
    algorithm_totals = work.groupby("algorithm_display", as_index=False)["count"].sum()
    algorithm_totals = algorithm_totals.sort_values(["count", "algorithm_display"], ascending=[True, True])
    algorithm_order = algorithm_totals["algorithm_display"].tolist()

    x_map = {stage: idx for idx, stage in enumerate(stage_order)}
    y_map = {algorithm: idx for idx, algorithm in enumerate(algorithm_order)}
    work["x_pos"] = work["stage"].map(x_map)
    work["y_pos"] = work["algorithm_display"].map(y_map)
    work["size"] = scale_bubble_sizes(work["count"], min_size=850, max_size=4200)

    fig, ax = plt.subplots(figsize=(11.2, 10.2), facecolor="white")
    ax.set_facecolor("white")
    scatter = ax.scatter(
        work["x_pos"],
        work["y_pos"],
        s=work["size"],
        c=work["count"],
        cmap="viridis",
        vmin=0,
        vmax=max(int(work["count"].max()), 1),
        alpha=0.82,
        edgecolors="none",
    )

    for _, row in work.iterrows():
        ax.text(
            row["x_pos"],
            row["y_pos"],
            str(int(row["count"])),
            ha="center",
            va="center",
            fontsize=12,
            color="white",
            fontweight="bold",
        )

    ax.set_xticks(range(len(stage_order)), [wrap_stage_label(stage) for stage in stage_order])
    ax.set_yticks(range(len(algorithm_order)), [wrap_algorithm_label(label) for label in algorithm_order])
    ax.set_xlabel("Clinical stage", fontsize=15)
    ax.set_ylabel("Algorithm family", fontsize=15)
    ax.tick_params(axis="both", labelsize=12, length=0)
    ax.grid(True, color="#e5e7eb", linewidth=1.0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(-0.55, len(stage_order) - 0.35)
    ax.set_ylim(-0.9, len(algorithm_order) - 0.2)

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.028, pad=0.03)
    cbar.ax.set_title("Count", fontsize=13, pad=10)
    cbar.ax.tick_params(labelsize=11)

    save_figure_variants(fig, out_stem)


def build_method_source_stage_heatmaps(
    counts_df: pd.DataFrame,
    out_stem: Path,
) -> None:
    work = counts_df.copy()
    work = work[work["count"] > 0].copy()
    if work.empty:
        return

    stage_order = [stage for stage in STAGE_ORDER if stage in work["stage"].unique() and stage != "Not specified"]
    modality_order = [
        modality
        for modality in MODALITY_ORDER
        if modality in work["source_modality_display"].unique()
    ]
    method_order = [
        method
        for method in METHOD_ORDER
        if method in work["ai_method_display"].unique()
    ]

    matrix_blocks = []
    xtick_positions: list[float] = []
    xtick_labels: list[str] = []
    stage_boundaries: list[float] = []
    stage_centers: list[tuple[float, str]] = []

    current_col = 0
    for stage in stage_order:
        subset = work[work["stage"] == stage]
        table = (
            subset.pivot(
                index="source_modality_display",
                columns="ai_method_display",
                values="count",
            )
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
    fig, ax = plt.subplots(figsize=(14.5, 6.8), facecolor="white")
    cmap = plt.cm.YlGnBu
    im = ax.imshow(full_matrix, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")

    for row_idx in range(full_matrix.shape[0]):
        for col_idx in range(full_matrix.shape[1]):
            value = int(full_matrix[row_idx, col_idx])
            label_color = "white" if value >= max(vmax * 0.45, 1) else "#16324f"
            ax.text(
                col_idx,
                row_idx,
                str(value),
                ha="center",
                va="center",
                fontsize=9.5,
                color=label_color,
                fontweight="bold" if value > 0 else None,
            )

    ax.set_xticks(xtick_positions, xtick_labels, rotation=35, ha="right")
    ax.set_yticks(range(len(modality_order)), modality_order)
    ax.tick_params(axis="both", labelsize=10.5, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, full_matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(modality_order), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for boundary in stage_boundaries:
        ax.axvline(boundary, color="#94a3b8", linewidth=1.7)

    for center, stage in stage_centers:
        ax.text(
            center,
            1.06,
            wrap_stage_label(stage).replace("\n", " "),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=12,
            color="#243b5a",
            fontweight="bold",
        )

    ax.set_xlabel("AI method grouped by clinical stage", fontsize=15, labelpad=20)
    ax.set_ylabel("Data source", fontsize=15)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.set_title("Count", fontsize=12, pad=8)
    cbar.ax.tick_params(labelsize=10.5)
    fig.subplots_adjust(left=0.17, right=0.9, bottom=0.22, top=0.88)
    save_figure_variants(fig, out_stem)


def collect_output_index() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for rq, folder in [
        ("RQ1", RQ1_OUTPUT_DIR),
        ("RQ1_supporting", RQ1_SUPPORTING_DIR),
        ("RQ2", RQ2_OUTPUT_DIR),
        ("RQ3", RQ3_OUTPUT_DIR),
        ("RQ3_supplement", RQ3_SUPPLEMENT_DIR),
    ]:
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                rows.append(
                    {
                        "rq": rq,
                        "relative_path": str(path.relative_to(BASE_DIR)),
                        "file_type": path.suffix.lstrip("."),
                    }
                )
    return rows


def refresh_master_denominator_table() -> None:
    frames = []
    for folder, filename in [
        (RQ1_OUTPUT_DIR, "rq1_denominators_q1_v2.csv"),
        (RQ2_OUTPUT_DIR, "rq2_denominators_q1_v2.csv"),
        (RQ3_OUTPUT_DIR, "rq3_denominators_q1_v2.csv"),
    ]:
        path = folder / filename
        if path.exists():
            frames.append(pd.read_csv(path))
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(MASTER_DENOMINATOR_FILE, index=False)


def run_rq1() -> None:
    import rq1_script

    ensure_dir(RQ1_OUTPUT_DIR)
    ensure_dir(RQ1_SUPPORTING_DIR)

    original_output_dir = rq1_script.OUTPUT_DIR
    original_support_dir = rq1_script.SUPPORTING_DIR
    try:
        rq1_script.OUTPUT_DIR = str(RQ1_OUTPUT_DIR.name)
        rq1_script.SUPPORTING_DIR = str(RQ1_SUPPORTING_DIR.name)
        rq1_script.main()
    finally:
        rq1_script.OUTPUT_DIR = original_output_dir
        rq1_script.SUPPORTING_DIR = original_support_dir

    algorithm_counts_path = RQ1_OUTPUT_DIR / "rq1_counts_algorithm_x_source_modality_by_stage.csv"
    if algorithm_counts_path.exists():
        algorithm_counts = pd.read_csv(algorithm_counts_path)
        build_reference_algorithm_bubble(
            counts_df=algorithm_counts,
            out_stem=RQ1_OUTPUT_DIR / "rq1_algorithm_bubbles_q1_v2",
        )

    method_counts_path = RQ1_OUTPUT_DIR / "rq1_counts_source_modality_x_method_by_stage.csv"
    if method_counts_path.exists():
        method_counts = pd.read_csv(method_counts_path)
        build_method_source_stage_heatmaps(
            counts_df=method_counts,
            out_stem=RQ1_OUTPUT_DIR / "rq1_method_source_stage_heatmap_q1_v2",
        )
        write_caption(
            RQ1_OUTPUT_DIR / "rq1_method_source_stage_heatmap_q1_v2_caption.txt",
            """
            RQ1 companion figure. Faceted heatmaps showing the distribution of AI methods across data sources within
            each functional stage of the clinical process in ASD. Each panel corresponds to one clinical stage; rows
            represent data sources, columns represent AI methods, cell color encodes the number of studies, and cell
            labels report counts directly. This figure complements the algorithm-by-stage bubble plot by covering the
            remaining RQ1 dimensions: AI methods, data sources, and clinical stages.
            """,
        )

    for stale_name in [
        "rq1_bubble_main_q1_v2.png",
        "rq1_bubble_main_q1_v2.pdf",
        "rq1_bubble_main_q1_v2.svg",
        "rq1_bubble_main_q1_v2_caption.txt",
        "rq1_method_source_stage_q1_v2.png",
        "rq1_method_source_stage_q1_v2.pdf",
        "rq1_method_source_stage_q1_v2.svg",
        "rq1_method_source_stage_q1_v2_caption.txt",
        "rq1_candidate_alluvial.png",
        "rq1_candidate_panel.png",
        "rq1_heatmap_algorithm_x_source_modality_by_stage.png",
        "rq1_heatmap_source_modality_x_method_by_stage.png",
    ]:
        stale_path = RQ1_OUTPUT_DIR / stale_name
        if stale_path.exists():
            stale_path.unlink()

    refresh_master_denominator_table()


def derive_q2_status(row: pd.Series) -> str:
    abs_signal = normalize_bool_signal(row.get("q2_candidate_abstract"))
    terms_signal = normalize_bool_signal(row.get("q2_candidate_terms"))
    if abs_signal is True or terms_signal is True:
        return "Positive"
    if abs_signal is False or terms_signal is False:
        return "Negative"
    return "Missing"


def derive_integration_approach_positive(row: pd.Series) -> str:
    stage = row["stage_norm"]
    title = norm_text(row.get("title"))
    abstract = norm_text(row.get("abstract"))
    ai_task = norm_text(row.get("AI_task_type"))
    notes = norm_text(row.get("notes_coding"))
    text = " ".join([title, abstract, ai_task, notes])

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
    if row["q2_status"] != "Positive":
        return "Not applicable"
    approach = row["integration_approach"]
    text = " ".join([norm_text(row.get("title")), norm_text(row.get("abstract")), norm_text(row.get("AI_task_type")), norm_text(row.get("notes_coding"))])
    if approach in {"Triage / questionnaires", "Mobile screening", "Feature extraction", "Risk stratification"}:
        return "Pre-decision"
    if approach == "Second-reader decision support":
        return "Pre-decision" if any(token in text for token in ["triage", "pre-read", "feature extraction"]) else "In-decision"
    if approach in {"Longitudinal dashboards", "Adaptive intervention", "Assistive tools"}:
        return "Post-decision"
    return "Unspecified"


def run_rq2() -> None:
    ensure_dir(RQ2_OUTPUT_DIR)

    df = normalize_common_fields(load_base_df())
    df["q2_abstract_bool"] = df["q2_candidate_abstract"].map(normalize_bool_signal)
    df["q2_terms_bool"] = df["q2_candidate_terms"].map(normalize_bool_signal)
    df["q2_status"] = df.apply(derive_q2_status, axis=1)
    df["integration_approach"] = df.apply(
        lambda row: derive_integration_approach_positive(row)
        if row["q2_status"] == "Positive"
        else f"{row['q2_status']} Q2 signal",
        axis=1,
    )
    df["decision_timing"] = df.apply(derive_decision_timing, axis=1)
    df["q2_signal_pattern"] = df.apply(
        lambda row: f"abstract={status_from_bool(row['q2_abstract_bool'])}; terms={status_from_bool(row['q2_terms_bool'])}",
        axis=1,
    )

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
        pd.crosstab(df["stage_norm"], df["integration_approach"])
        .reindex(index=ordered_categories(df["stage_norm"].unique().tolist(), STAGE_ORDER), columns=integration_order, fill_value=0)
    )
    timing_table = (
        pd.crosstab(df["q2_status"], df["decision_timing"])
        .reindex(index=["Positive", "Negative", "Missing"], columns=["Pre-decision", "In-decision", "Post-decision", "Unspecified", "Not applicable"], fill_value=0)
    )

    stage_integration.to_csv(RQ2_OUTPUT_DIR / "rq2_stage_x_integration_q1_v2.csv")
    timing_table.to_csv(RQ2_OUTPUT_DIR / "rq2_timing_table_q1_v2.csv")
    df.to_csv(RQ2_OUTPUT_DIR / "rq2_tripartite_dataset_q1_v2.csv", index=False)

    review_rows = df[
        df["integration_approach"].isin(["Unspecified integration", "Missing Q2 signal"])
        | df["decision_timing"].eq("Unspecified")
    ].copy()
    review_rows.to_csv(RQ2_OUTPUT_DIR / "rq2_review_rows_q1_v2.csv", index=False)

    denominator_rows = [
        {
            "rq": "RQ2",
            "subset": "all_rows",
            "group": "All studies",
            "denominator_n": int(len(df)),
            "notes": "Unique studies in the consolidated sheet.",
        }
    ]
    for status, count in df["q2_status"].value_counts(dropna=False).reindex(["Positive", "Negative", "Missing"], fill_value=0).items():
        denominator_rows.append(
            {
                "rq": "RQ2",
                "subset": "q2_status",
                "group": status,
                "denominator_n": int(count),
                "notes": "Derived from q2_candidate_abstract and q2_candidate_terms without hiding missing values.",
            }
        )
    pd.DataFrame(denominator_rows).to_csv(RQ2_OUTPUT_DIR / "rq2_denominators_q1_v2.csv", index=False)

    fig, heat_ax = plt.subplots(figsize=(13.5, 6.8), facecolor="white")

    heat_values = stage_integration.to_numpy(dtype=float)
    heat_ax.imshow(heat_values, cmap="Blues", aspect="auto")
    row_totals = heat_values.sum(axis=1)
    for row_idx in range(heat_values.shape[0]):
        for col_idx in range(heat_values.shape[1]):
            value = int(heat_values[row_idx, col_idx])
            share = 0.0 if row_totals[row_idx] == 0 else value / row_totals[row_idx]
            label = "0" if value == 0 else f"{value}\n{share:.0%}"
            heat_ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=8)
    heat_ax.set_xticks(range(len(stage_integration.columns)), stage_integration.columns, rotation=40, ha="right")
    heat_ax.set_yticks(range(len(stage_integration.index)), stage_integration.index)
    heat_ax.set_xlabel("Integration approach")
    heat_ax.set_ylabel("Clinical stage")
    save_figure_variants(fig, RQ2_OUTPUT_DIR / "rq2_heatmap_and_timing_q1_v2")
    write_caption(
        RQ2_OUTPUT_DIR / "rq2_heatmap_and_timing_q1_v2_caption.txt",
        """
        RQ2. Stage-by-integration heatmap. The heatmap shows study counts by primary clinical stage and derived
        integration approach. Positive Q2 studies are assigned to workflow-oriented integration categories using keyword
        and stage fallback rules. Cell annotations report the count and the within-stage percentage. Study-level
        classifications are stored in rq2_tripartite_dataset_q1_v2.csv and denominators are listed in
        rq2_denominators_q1_v2.csv.
        """,
    )

    refresh_master_denominator_table()


def q3_status(value: object) -> str:
    return status_from_bool(normalize_bool_signal(value))


def build_q3_combo_summary(df: pd.DataFrame) -> pd.DataFrame:
    work = normalize_common_fields(df)
    for col in PRACTICE_ORDER:
        work[f"{col}_status"] = work[col].map(q3_status)

    combo_rows = []
    combo_cols = ["stage_norm", "modality_norm", "method_norm"]
    for combo_values, group in work.groupby(combo_cols, dropna=False):
        row = {
            "stage_norm": combo_values[0],
            "modality_norm": combo_values[1],
            "method_norm": combo_values[2],
            "combo_n": int(len(group)),
        }
        for col in PRACTICE_ORDER:
            counts = group[f"{col}_status"].value_counts().reindex(["Positive", "Negative", "Missing"], fill_value=0)
            row[f"{col}_positive"] = int(counts["Positive"])
            row[f"{col}_negative"] = int(counts["Negative"])
            row[f"{col}_missing"] = int(counts["Missing"])
            row[f"{col}_positive_rate"] = counts["Positive"] / len(group)
        combo_rows.append(row)

    combo = pd.DataFrame(combo_rows)
    combo["combo_label"] = combo.apply(
        lambda row: f"{row['stage_norm']} | {row['modality_norm']} | {row['method_norm']} (n={int(row['combo_n'])})",
        axis=1,
    )
    combo = combo.sort_values(["combo_n", "stage_norm", "modality_norm", "method_norm"], ascending=[False, True, True, True]).reset_index(drop=True)
    return combo


def run_rq3() -> None:
    ensure_dir(RQ3_OUTPUT_DIR)
    ensure_dir(RQ3_SUPPLEMENT_DIR)

    df = normalize_common_fields(load_base_df())
    global_rows = []
    for col, label in PRACTICE_COLS.items():
        statuses = df[col].map(q3_status)
        counts = statuses.value_counts().reindex(["Positive", "Negative", "Missing"], fill_value=0)
        denominator = int(counts.sum())
        global_rows.append(
            {
                "practice_signal": col,
                "practice": label,
                "positive_n": int(counts["Positive"]),
                "negative_n": int(counts["Negative"]),
                "missing_n": int(counts["Missing"]),
                "total_n": denominator,
                "positive_rate": counts["Positive"] / denominator if denominator else math.nan,
            }
        )
    global_summary = pd.DataFrame(global_rows)
    global_summary.to_csv(RQ3_OUTPUT_DIR / "rq3_global_practice_summary_q1_v2.csv", index=False)

    combo_summary = build_q3_combo_summary(df)
    combo_summary.to_csv(RQ3_SUPPLEMENT_DIR / "rq3_combo_summary_q1_v2.csv", index=False)
    combo_summary[combo_summary["combo_n"] < 5].to_csv(
        RQ3_SUPPLEMENT_DIR / "rq3_combos_n_lt_5_q1_v2.csv",
        index=False,
    )

    denominator_rows = [
        {
            "rq": "RQ3",
            "subset": "all_rows",
            "group": "All studies",
            "denominator_n": int(len(df)),
            "notes": "All studies were retained and missing practice coding remains visible.",
        }
    ]
    for _, row in global_summary.iterrows():
        denominator_rows.append(
            {
                "rq": "RQ3",
                "subset": "practice",
                "group": row["practice"],
                "denominator_n": int(row["total_n"]),
                "notes": f"Positive={int(row['positive_n'])}; Negative={int(row['negative_n'])}; Missing={int(row['missing_n'])}.",
            }
        )
    pd.DataFrame(denominator_rows).to_csv(RQ3_OUTPUT_DIR / "rq3_denominators_q1_v2.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 6.5), facecolor="white")
    x = np.arange(len(global_summary))
    bottom = np.zeros(len(global_summary))
    for status in ["Positive", "Negative", "Missing"]:
        values = global_summary[f"{status.lower()}_n"].to_numpy(dtype=float)
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=STATUS_COLORS[status],
            edgecolor="white",
            linewidth=0.8,
            label=status,
        )
        for idx, value in enumerate(values):
            if value > 0:
                ax.text(x[idx], bottom[idx] + value / 2, f"{int(value)}", ha="center", va="center", fontsize=9, color="white" if status != "Missing" else "black")
        bottom += values
    ax.set_xticks(x, global_summary["practice"], rotation=15, ha="right")
    ax.set_ylabel("Study counts")
    ax.set_title("Global methodological practices across all studies", fontsize=14, fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.08))
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    for idx, row in global_summary.iterrows():
        ax.text(x[idx], row["total_n"] + 2.5, f"{row['positive_rate']:.0%} positive", ha="center", va="bottom", fontsize=9)
    save_figure_variants(fig, RQ3_OUTPUT_DIR / "rq3_global_practices_q1_v2")

    supplement_df = combo_summary[combo_summary["combo_n"] >= 5].copy()
    if not supplement_df.empty:
        rate_cols = [f"{col}_positive_rate" for col in PRACTICE_ORDER]
        rate_matrix = supplement_df[rate_cols].to_numpy(dtype=float)
        fig2_height = max(5.5, 0.42 * len(supplement_df) + 1.8)
        fig2, ax2 = plt.subplots(figsize=(11, fig2_height), facecolor="white")
        im = ax2.imshow(rate_matrix, cmap="Greens", vmin=0, vmax=1, aspect="auto")
        for row_idx in range(rate_matrix.shape[0]):
            for col_idx, practice_col in enumerate(PRACTICE_ORDER):
                count = int(supplement_df.iloc[row_idx][f"{practice_col}_positive"])
                total = int(supplement_df.iloc[row_idx]["combo_n"])
                value = rate_matrix[row_idx, col_idx]
                label = f"{count}/{total}"
                ax2.text(col_idx, row_idx, label, ha="center", va="center", fontsize=8)
        ax2.set_xticks(range(len(PRACTICE_ORDER)), [PRACTICE_COLS[col] for col in PRACTICE_ORDER], rotation=20, ha="right")
        ax2.set_yticks(range(len(supplement_df)), supplement_df["combo_label"])
        ax2.set_title("Supplement: combinations with n >= 5", fontsize=13, fontweight="bold")
        fig2.colorbar(im, ax=ax2, fraction=0.03, pad=0.02, label="Positive rate")
        save_figure_variants(fig2, RQ3_SUPPLEMENT_DIR / "rq3_combo_heatmap_n_ge_5_q1_v2")

    write_caption(
        RQ3_OUTPUT_DIR / "rq3_global_practices_q1_v2_caption.txt",
        """
        RQ3. Global overview of four methodological practices across the full study set. Each stacked bar shows the
        number of studies coded as Positive, Negative, or Missing for external validation, multisource integration,
        model explainability, and cross-site robustness. The percentage printed above each bar is the Positive share
        using the full practice-specific denominator. Combination-level analyses were moved out of the main figure; all
        combination summaries are stored in rq3_results_q1_v2/supplement, and combinations with n < 5 are exported to
        rq3_combos_n_lt_5_q1_v2.csv for supplementary reporting.
        """,
    )

    refresh_master_denominator_table()


def run_all() -> None:
    run_rq1()
    run_rq2()
    run_rq3()
    refresh_master_denominator_table()
    index_rows = collect_output_index()
    if index_rows:
        pd.DataFrame(index_rows).to_csv(BASE_DIR / "q1_v2_output_index.csv", index=False)


def run_script_entry(target: str) -> None:
    script = BASE_DIR / target
    subprocess.run([sys.executable, str(script)], check=True)
