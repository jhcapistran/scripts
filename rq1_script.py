"""
RQ1 plots generator (no argparse).

This version generates only the stage-resolved outputs that directly answer
the RQ1 wording about the distribution of AI methods and algorithms across
source modalities and clinical stages.

Edit the CONFIG section only.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rich.console import Console
from rich.table import Table


# ----------------------------
# CONFIGURATION (EDIT ONLY THIS)
# ----------------------------
INPUT_FILES = [
    {"path": "aceptados_300_v2_codificado_RA.xlsx", "reviewer": "Revisor A"},
    {"path": "aceptados_300_codificado_RB.xlsx", "reviewer": "Revisor B"},
]
SHEET_NAME = 0  # 0 = first sheet
OUTPUT_DIR = "rq1_results"
SUPPORTING_DIR = "rq1_supporting"

MODALITY_COL = "modalidad"
STAGE_COL = "stage_primary"
AI_METHOD_COL = "tipo_IA"
ALGORITHM_COL = "AI_algorithm_main"
NOTES_COL = "notes_coding"

STAGE_FLAG_COLS = {
    "stage_prescreening": "Prescreening",
    "stage_screening": "Screening",
    "stage_diagnosis": "Diagnosis",
    "stage_prognosis": "Prognosis",
    "stage_monitoring_intervention": "Monitoring/intervention",
}

ALGORITHM_TOPK = 8
PAPER_ALGORITHM_TOPK = 5
# ----------------------------


STAGE_ORDER = [
    "Prescreening",
    "Screening",
    "Diagnosis",
    "Prognosis",
    "Monitoring/intervention",
]
STAGE_DISPLAY = {
    "Prescreening": "Prescreening",
    "Screening": "Screening",
    "Diagnosis": "Diagnosis",
    "Prognosis": "Prognosis",
    "Monitoring/intervention": "Monitoring<br>intervention",
}
MODALITY_ORDER = [
    "Image",
    "Physiological signals",
    "Text / NLP",
    "Audio / Voice",
    "Multimodal",
    "Not specified",
]
METHOD_ORDER = [
    "Machine Learning",
    "Deep Learning",
    "Hybrid",
    "Not specified",
]
OTHER_ALGORITHM_LABEL = "Other algorithms"
UNCLEAR_ALGORITHM_LABEL = "Other / unclear"

MODALITY_TRANSLATIONS = {
    "imagen": "Image",
    "señales fisiológicas": "Physiological signals",
    "senales fisiologicas": "Physiological signals",
    "seã±ales fisiolã³gicas": "Physiological signals",
    "texto · nlp": "Text / NLP",
    "texto / nlp": "Text / NLP",
    "texto â· nlp": "Text / NLP",
    "audio · voz": "Audio / Voice",
    "audio / voz": "Audio / Voice",
    "audio â· voz": "Audio / Voice",
    "multimodal": "Multimodal",
    "no especificado": "Not specified",
    "no especificada": "Not specified",
}
PRIMARY_STAGE_TRANSLATIONS = {
    "prescreening": "Prescreening",
    "screening": "Screening",
    "diagnosis": "Diagnosis",
    "prognosis": "Prognosis",
    "monitoring/intervention": "Monitoring/intervention",
    "monitoring_intervention": "Monitoring/intervention",
    "not clear": "Not clear",
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
UNCLEAR_ALGORITHM_KEYS = {
    "other / not clear",
    "other/not clear",
    "not clear",
    "other",
    "no especificado",
    "not specified",
}

CORE_OUTPUT_FILES = {
    "rq1_counts_source_modality_x_method_by_stage.csv",
    "rq1_counts_algorithm_x_source_modality_by_stage.csv",
    "rq1_heatmap_source_modality_x_method_by_stage.png",
    "rq1_heatmap_algorithm_x_source_modality_by_stage.png",
    "rq1_algorithm_plot_mapping.csv",
    "rq1_candidate_alluvial.png",
    "rq1_candidate_panel.png",
}

SUPPORTING_OUTPUT_FILES = {
    "rq1_notes_summary.txt",
    "rq1_notes_full_context.csv",
}

METHOD_COLORS = {
    "Machine Learning": "#1f77b4",
    "Deep Learning": "#d62728",
    "Hybrid": "#2ca02c",
    "Not specified": "#7f7f7f",
}
STAGE_COLORS = {
    "Prescreening": "#4c78a8",
    "Screening": "#f58518",
    "Diagnosis": "#54a24b",
    "Prognosis": "#e45756",
    "Monitoring/intervention": "#72b7b2",
}
NEUTRAL_NODE_COLOR = "#d9dde3"
PLOT_LABELS = {
    "Physiological signals": "Physiological<br>signals",
    "Monitoring/intervention": "Monitoring<br>intervention",
    "Logistic Regression": "Logistic<br>Regression",
    "Clustering (k-means / GMM / ...)": "Clustering",
    "XGBoost / GBDT": "Gradient<br>boosting",
    "Other algorithms": "Other<br>algorithms",
}


# ----------------------------
# Raw value helpers
# ----------------------------
def clean_cell_value(x: object) -> object:
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()
    return s if s else pd.NA


def unify_case_variants(series: pd.Series) -> pd.Series:
    non_null = series.dropna()
    if non_null.empty:
        return series

    variant_counts: dict[str, Counter[str]] = {}
    for value in non_null.astype(str):
        key = value.casefold()
        variant_counts.setdefault(key, Counter())[value] += 1

    canonical_map: dict[str, str] = {}
    for key, counts in variant_counts.items():
        canonical_map[key] = sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0]

    return series.map(lambda x: canonical_map.get(str(x).casefold(), x) if pd.notna(x) else x)


def translate_to_english(series: pd.Series, mapping: dict[str, str]) -> pd.Series:
    return series.map(
        lambda x: mapping.get(str(x).casefold(), x) if pd.notna(x) else x
    )


def ordered_categories(observed: list[str], preferred: list[str]) -> list[str]:
    ordered = [x for x in preferred if x in observed]
    extras = sorted(x for x in observed if x not in preferred)
    return ordered + extras


def flag_is_positive(x: object) -> bool:
    try:
        return float(x) > 0
    except (TypeError, ValueError):
        return False


def normalize_stage_value(x: object) -> object:
    value = clean_cell_value(x)
    if pd.isna(value):
        return pd.NA
    return PRIMARY_STAGE_TRANSLATIONS.get(str(value).casefold(), value)


def normalize_algorithm_value(x: object) -> str:
    value = clean_cell_value(x)
    if pd.isna(value):
        return UNCLEAR_ALGORITHM_LABEL
    s = str(value)
    return UNCLEAR_ALGORITHM_LABEL if s.casefold() in UNCLEAR_ALGORITHM_KEYS else s


def extract_stage_assignments(row: pd.Series) -> tuple[list[str], str]:
    stages = [label for col, label in STAGE_FLAG_COLS.items() if flag_is_positive(row.get(col))]
    if stages:
        return stages, "stage_flags"

    primary = normalize_stage_value(row.get(STAGE_COL))
    if pd.notna(primary) and primary in STAGE_ORDER:
        return [str(primary)], "stage_primary_fallback"

    return [], "unresolved"


def compress_algorithm_labels(series: pd.Series, topk: int) -> tuple[pd.Series, list[str]]:
    specific = series[series != UNCLEAR_ALGORITHM_LABEL]
    top_specific = specific.value_counts().head(topk).index.tolist()

    grouped = series.map(
        lambda x: x
        if x in top_specific or x == UNCLEAR_ALGORITHM_LABEL
        else OTHER_ALGORITHM_LABEL
    )

    order = list(top_specific)
    if (grouped == OTHER_ALGORITHM_LABEL).any():
        order.append(OTHER_ALGORITHM_LABEL)
    if (grouped == UNCLEAR_ALGORITHM_LABEL).any():
        order.append(UNCLEAR_ALGORITHM_LABEL)

    return grouped, order


def keep_all_algorithm_labels(series: pd.Series) -> tuple[pd.Series, list[str]]:
    specific = series[series != UNCLEAR_ALGORITHM_LABEL]
    order = specific.value_counts().index.tolist()
    if (series == UNCLEAR_ALGORITHM_LABEL).any():
        order.append(UNCLEAR_ALGORITHM_LABEL)
    return series.copy(), order


# ----------------------------
# Plotting helpers (matplotlib only)
# ----------------------------
def draw_heatmap(
    ax: plt.Axes,
    table: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    cmap: str,
    vmax: int,
) -> plt.AxesImage:
    im = ax.imshow(table.values, aspect="auto", cmap=cmap, vmin=0, vmax=max(vmax, 1))

    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_xticklabels(table.columns, rotation=35, ha="right")
    ax.set_yticklabels(table.index)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)

    ax.set_xticks(np.arange(-0.5, table.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, table.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    thresh = max(vmax, 1) * 0.55
    fontsize = 8 if max(table.shape) >= 9 else 9
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            val = int(table.iloc[i, j])
            ax.text(
                j,
                i,
                str(val),
                ha="center",
                va="center",
                fontsize=fontsize,
                color="white" if val >= thresh else "black",
            )

    return im


def save_faceted_heatmaps(
    tables_by_stage: dict[str, pd.DataFrame],
    outpath: Path,
    xlabel: str,
    ylabel: str,
    cmap: str = "Blues",
) -> None:
    used_tables = [table for table in tables_by_stage.values() if not table.empty]
    if not used_tables:
        return

    max_rows = max(table.shape[0] for table in used_tables)
    max_cols = max(table.shape[1] for table in used_tables)
    vmax = max(int(table.to_numpy().max()) for table in used_tables)

    n_plots = len(STAGE_ORDER)
    ncols = 3
    nrows = math.ceil(n_plots / ncols)
    fig_w = max(16, ncols * (max_cols * 1.2 + 2.4))
    fig_h = max(9, nrows * (max_rows * 0.55 + 2.8))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        constrained_layout=True,
    )
    flat_axes = axes.flatten()
    last_im = None

    for idx, stage in enumerate(STAGE_ORDER):
        ax = flat_axes[idx]
        table = tables_by_stage[stage]
        title = f"{stage}\n(n={int(table.to_numpy().sum())})"
        last_im = draw_heatmap(
            ax=ax,
            table=table,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            cmap=cmap,
            vmax=vmax,
        )

    for idx in range(n_plots, len(flat_axes)):
        flat_axes[idx].axis("off")

    if last_im is not None:
        fig.colorbar(last_im, ax=flat_axes[:n_plots], fraction=0.02, pad=0.02)

    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def rgba_from_hex(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def plot_label(label: str) -> str:
    return PLOT_LABELS.get(label, label)


def sorted_flow_df(flow_df: pd.DataFrame, dim_orders: dict[str, list[str]]) -> pd.DataFrame:
    ordered = flow_df.copy()
    sort_cols: list[str] = []
    for dim, order in dim_orders.items():
        order_map = {value: idx for idx, value in enumerate(order)}
        sort_col = f"_{dim}_sort"
        ordered[sort_col] = ordered[dim].map(order_map).fillna(len(order_map))
        sort_cols.append(sort_col)
    return ordered.sort_values(sort_cols + ["count"], ascending=[True] * len(sort_cols) + [False])


def build_sankey_trace(
    flow_df: pd.DataFrame,
    dim_orders: dict[str, list[str]],
    link_color_dim: str | None = None,
) -> go.Sankey:
    dims = list(dim_orders.keys())
    present_by_dim = {
        dim: [value for value in order if value in set(flow_df[dim].dropna().astype(str))]
        for dim, order in dim_orders.items()
    }

    node_index: dict[tuple[str, str], int] = {}
    node_labels: list[str] = []
    node_colors: list[str] = []
    node_x: list[float] = []
    node_y: list[float] = []

    x_positions = np.linspace(0.03, 0.97, len(dims))
    for x, dim in zip(x_positions, dims):
        values = present_by_dim[dim]
        y_positions = [0.5] if len(values) == 1 else np.linspace(0.04, 0.96, len(values))
        for y, value in zip(y_positions, values):
            node_index[(dim, value)] = len(node_labels)
            node_labels.append(plot_label(value))
            if dim == "ai_method_display":
                node_colors.append(METHOD_COLORS.get(value, NEUTRAL_NODE_COLOR))
            elif dim == "stage":
                node_colors.append(STAGE_COLORS.get(value, NEUTRAL_NODE_COLOR))
            else:
                node_colors.append(NEUTRAL_NODE_COLOR)
            node_x.append(float(x))
            node_y.append(float(y))

    ordered_flow = sorted_flow_df(flow_df, dim_orders)
    link_buckets: dict[tuple[int, int, str], int] = {}

    for row in ordered_flow.itertuples(index=False):
        row_dict = row._asdict()
        if link_color_dim is None:
            color = rgba_from_hex("#9aa3ad", 0.28)
        else:
            color_key = str(row_dict[link_color_dim])
            color = rgba_from_hex(METHOD_COLORS.get(color_key, "#9aa3ad"), 0.30)
        for left_dim, right_dim in zip(dims, dims[1:]):
            source = node_index[(left_dim, str(row_dict[left_dim]))]
            target = node_index[(right_dim, str(row_dict[right_dim]))]
            bucket_key = (source, target, color)
            link_buckets[bucket_key] = link_buckets.get(bucket_key, 0) + int(row_dict["count"])

    sources: list[int] = []
    targets: list[int] = []
    values: list[int] = []
    colors: list[str] = []
    for (source, target, color), value in link_buckets.items():
        sources.append(source)
        targets.append(target)
        values.append(value)
        colors.append(color)

    return go.Sankey(
        arrangement="fixed",
        node=dict(
            pad=16,
            thickness=18,
            line=dict(color="white", width=0.8),
            label=node_labels,
            color=node_colors,
            x=node_x,
            y=node_y,
            hovertemplate="%{label}<extra></extra>",
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=colors,
            hovertemplate="Count: %{value}<extra></extra>",
        ),
    )


def save_candidate_alluvial(
    flow_df: pd.DataFrame,
    modality_order: list[str],
    method_order: list[str],
    algorithm_order: list[str],
    outpath: Path,
) -> None:
    dim_orders = {
        "stage": STAGE_ORDER,
        "source_modality_display": modality_order,
        "ai_method_display": method_order,
        "algorithm_paper": algorithm_order,
    }
    trace = build_sankey_trace(
        flow_df=flow_df,
        dim_orders=dim_orders,
        link_color_dim="ai_method_display",
    )
    fig = go.Figure(trace)
    fig.update_layout(
        width=1800,
        height=950,
        font=dict(size=11, family="Arial"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=40, r=40, t=90, b=30),
    )
    for x, label in zip(
        np.linspace(0.03, 0.97, 4),
        ["Clinical stage", "Source modality", "AI method", "Algorithm family"],
    ):
        fig.add_annotation(
            x=x,
            y=1.08,
            xref="paper",
            yref="paper",
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=12),
        )
    fig.write_image(outpath, scale=2)


def scale_bubble_sizes(
    counts: pd.Series,
    min_size: float = 50,
    max_size: float = 92,
) -> pd.Series:
    if counts.empty:
        return counts.astype(float)

    counts = counts.astype(float)
    max_count = counts.max()
    if max_count <= 0:
        return pd.Series(min_size, index=counts.index, dtype=float)

    normalized = np.sqrt(counts / max_count)
    return min_size + normalized * (max_size - min_size)


def save_candidate_panel(
    flow_df: pd.DataFrame,
    modality_order: list[str],
    method_order: list[str],
    algorithm_order: list[str],
    outpath: Path,
) -> None:
    stage_labels = [STAGE_DISPLAY.get(stage, stage) for stage in STAGE_ORDER]
    modality_labels = [plot_label(value) for value in modality_order]

    fig = make_subplots(
        rows=3,
        cols=1,
        specs=[[{"type": "xy"}], [{"type": "heatmap"}], [{"type": "xy"}]],
        row_heights=[0.24, 0.30, 0.46],
        vertical_spacing=0.10,
        subplot_titles=(
            "A. AI method mix within each clinical stage",
            "B. Source modalities used to support decisions at each stage",
            "C. Algorithm families most frequently used across stages",
        ),
    )

    stage_method = (
        flow_df.groupby(["stage", "ai_method_display"])["count"]
        .sum()
        .reset_index()
    )
    stage_totals = (
        stage_method.groupby("stage")["count"]
        .sum()
        .reindex(STAGE_ORDER, fill_value=0)
    )
    for method in method_order:
        subset = stage_method[stage_method["ai_method_display"] == method]
        counts_map = {
            row["stage"]: int(row["count"])
            for _, row in subset.iterrows()
        }
        counts = [counts_map.get(stage, 0) for stage in STAGE_ORDER]
        fig.add_trace(
            go.Bar(
                x=stage_labels,
                y=counts,
                name=method,
                marker_color=METHOD_COLORS.get(method, "#9aa3ad"),
                text=[str(v) if v > 0 else "" for v in counts],
                textposition="inside",
                textfont=dict(color="white", size=11),
                hovertemplate=(
                    "<b>%{x}</b><br>AI method: "
                    + method
                    + "<br>Count: %{y}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=stage_labels,
            y=stage_totals.tolist(),
            mode="text",
            text=[f"n={int(v)}" if v > 0 else "" for v in stage_totals.tolist()],
            textposition="top center",
            textfont=dict(color="#23364d", size=12),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    stage_modality = (
        flow_df.groupby(["source_modality_display", "stage"])["count"]
        .sum()
        .reset_index()
    )
    modality_matrix = (
        stage_modality.pivot(
            index="source_modality_display",
            columns="stage",
            values="count",
        )
        .reindex(index=modality_order, columns=STAGE_ORDER, fill_value=0)
        .astype(int)
    )
    fig.add_trace(
        go.Heatmap(
            z=modality_matrix.to_numpy(),
            x=stage_labels,
            y=modality_labels,
            text=modality_matrix.to_numpy(),
            texttemplate="%{text}",
            textfont=dict(size=11),
            colorscale=[
                [0.0, "#f3efe6"],
                [0.25, "#f0c987"],
                [0.50, "#df8f44"],
                [0.75, "#b75b31"],
                [1.0, "#7a2e1f"],
            ],
            zmin=0,
            zmax=float(modality_matrix.to_numpy().max()) if not modality_matrix.empty else 1.0,
            colorbar=dict(
                title="Studies",
                thickness=16,
                len=0.28,
                x=1.01,
                y=0.515,
            ),
            hovertemplate="<b>%{y}</b><br>Stage: %{x}<br>Count: %{z}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    algorithm_totals = (
        flow_df.groupby("algorithm_paper")["count"]
        .sum()
        .sort_values(ascending=False)
    )
    top_specific_algorithms = [
        alg
        for alg in algorithm_totals.index
        if alg != UNCLEAR_ALGORITHM_LABEL
    ][:10]
    panel_algorithm_order = top_specific_algorithms.copy()
    if UNCLEAR_ALGORITHM_LABEL in algorithm_totals.index:
        panel_algorithm_order.append(UNCLEAR_ALGORITHM_LABEL)

    alg_stage = (
        flow_df[flow_df["algorithm_paper"].isin(panel_algorithm_order)]
        .groupby(["algorithm_paper", "stage"])["count"]
        .sum()
        .reset_index()
    )
    alg_stage["stage_plot"] = alg_stage["stage"].map(lambda value: STAGE_DISPLAY.get(value, value))
    alg_stage["algorithm_plot"] = alg_stage["algorithm_paper"].map(plot_label)
    alg_stage["marker_size"] = scale_bubble_sizes(alg_stage["count"], min_size=18, max_size=56)
    alg_stage["label"] = alg_stage["count"].map(lambda value: str(int(value)) if value >= 3 else "")
    y_order = [plot_label(value) for value in panel_algorithm_order]
    fig.add_trace(
        go.Scatter(
            x=alg_stage["stage_plot"],
            y=alg_stage["algorithm_plot"],
            mode="markers+text",
            text=alg_stage["label"],
            customdata=alg_stage["count"],
            textposition="middle center",
            textfont=dict(color="white", size=11, family="Arial Black"),
            marker=dict(
                size=alg_stage["marker_size"],
                color=alg_stage["count"],
                colorscale="magma",
                cmin=0,
                cmax=float(alg_stage["count"].max()) if not alg_stage.empty else 1.0,
                opacity=0.90,
                line=dict(color="white", width=1.2),
                showscale=False,
            ),
            hovertemplate="<b>%{y}</b><br>Stage: %{x}<br>Count: %{customdata}<extra></extra>",
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    fig.update_layout(
        barmode="stack",
        width=1250,
        height=1280,
        font=dict(size=13, family="Arial"),
        paper_bgcolor="#f5f1e8",
        plot_bgcolor="#fffaf2",
        margin=dict(l=80, r=80, t=120, b=60),
        legend=dict(
            title="AI method",
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0.0,
        ),
        title=dict(
            text=(
                "RQ1. How AI approaches are integrated across the clinical workflow in ASD"
                "<br><sup>Stage volume, source modality, and dominant algorithm families</sup>"
            ),
            x=0.5,
        ),
        uniformtext_minsize=10,
        uniformtext_mode="hide",
    )

    fig.update_xaxes(
        categoryorder="array",
        categoryarray=stage_labels,
        showgrid=False,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Number of stage assignments",
        showgrid=True,
        gridcolor="#e2d8c8",
        zeroline=False,
        row=1,
        col=1,
    )

    fig.update_xaxes(
        categoryorder="array",
        categoryarray=stage_labels,
        tickangle=0,
        row=2,
        col=1,
    )
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=list(reversed(modality_labels)),
        row=2,
        col=1,
    )

    fig.update_xaxes(
        title_text="Clinical stage",
        categoryorder="array",
        categoryarray=stage_labels,
        showgrid=True,
        gridcolor="#e2d8c8",
        zeroline=False,
        row=3,
        col=1,
    )
    fig.update_yaxes(
        title_text="Algorithm family",
        categoryorder="array",
        categoryarray=list(reversed(y_order)),
        showgrid=True,
        gridcolor="#e2d8c8",
        zeroline=False,
        row=3,
        col=1,
    )

    fig.add_annotation(
        x=0.5,
        y=0.705,
        xref="paper",
        yref="paper",
        text="Heatmap cells show counts of studies by source modality within each clinical stage",
        showarrow=False,
        font=dict(size=11, color="#5a4630"),
    )
    fig.add_annotation(
        x=0.5,
        y=0.02,
        xref="paper",
        yref="paper",
        text="Bubble size and color encode frequency; labels are shown for counts >= 3",
        showarrow=False,
        font=dict(size=11, color="#5a4630"),
    )

    fig.write_image(outpath, scale=2)


console = Console()

NOTE_TAG_RULES: list[tuple[str, tuple[str, ...]]] = [
    (
        "Algorithm not explicit or unspecified",
        (
            r"no explícito",
            r"no explicito",
            r"no especificado",
            r"not specified",
            r"not explicit",
        ),
    ),
    (
        "Needs full-text confirmation",
        (
            r"full[- ]text",
            r"confirmar",
        ),
    ),
    (
        "Outside codebook or forced mapping",
        (
            r"fuera del codebook",
            r"outside (the )?codebook",
            r"\bmapped to\b",
            r"\bmapea\b",
            r"allowed categories",
        ),
    ),
    (
        "Multiple models compared",
        (
            r"compara",
            r"comparative",
            r"varios modelos",
            r"multiple .* models",
            r"17 supervised models",
        ),
    ),
    (
        "Primary model chosen from comparisons",
        (
            r"\bmejor\b",
            r"\bbest\b",
            r"highest test accuracy",
            r"reported as highest",
        ),
    ),
    (
        "Hybrid or fusion architecture",
        (
            r"hybrid",
            r"fusion",
            r"late-fusion",
            r"cnn\+rnn",
            r"\bvit\b",
            r"\blstm\b",
            r"\bgru\b",
        ),
    ),
    (
        "Explainability or validation detail",
        (
            r"\bshap\b",
            r"5-fold",
            r"cross-validation",
            r"explainability",
            r"\bcv\b",
        ),
    ),
]


def classify_note_tags(note: str) -> list[str]:
    tags: list[str] = []
    for label, patterns in NOTE_TAG_RULES:
        if any(re.search(pattern, note, flags=re.IGNORECASE) for pattern in patterns):
            tags.append(label)
    return tags


def build_notes_support_df(df: pd.DataFrame) -> pd.DataFrame:
    if NOTES_COL not in df.columns:
        return pd.DataFrame()

    notes_df = df.copy()
    notes_df[NOTES_COL] = notes_df[NOTES_COL].fillna("").astype(str).str.strip()
    notes_df = notes_df[notes_df[NOTES_COL].astype(bool)].copy()
    if notes_df.empty:
        return notes_df

    notes_df["stage_assignments_display"] = notes_df["stage_assignments"].map(
        lambda values: " | ".join(values) if values else ""
    )
    notes_df["notes_issue_tags"] = notes_df[NOTES_COL].map(
        lambda note: " | ".join(classify_note_tags(note))
    )

    preferred_cols = [
        "study_id",
        "title",
        "year",
        "doi",
        "assigned_to",
        "source_file",
        "source_modality_display",
        "stage_assignments_display",
        "stage_assignment_origin",
        "ai_method_display",
        "algorithm_display_raw",
        "notes_issue_tags",
        NOTES_COL,
    ]
    existing_cols = [col for col in preferred_cols if col in notes_df.columns]
    return notes_df[existing_cols].sort_values(
        by=[col for col in ["assigned_to", "year", "study_id", "title"] if col in notes_df.columns],
        na_position="last",
    )


def run_notes_audit(df: pd.DataFrame, outdir: Path) -> None:
    notes_df = build_notes_support_df(df)
    if notes_df.empty:
        return

    notes_df.to_csv(outdir / "rq1_notes_full_context.csv", index=False)

    tag_counts: list[tuple[str, int]] = []
    for label, _ in NOTE_TAG_RULES:
        count = int(notes_df["notes_issue_tags"].str.contains(label, regex=False).sum())
        if count > 0:
            tag_counts.append((label, count))

    summary_lines = [
        "RQ1 notes support summary",
        "Purpose: preserve the full narrative notes used to document coding decisions,",
        "algorithm adjudication, and methodological clarification for the paper.",
        "",
        f"Notes column: {NOTES_COL}",
        f"Total analyzed rows: {len(df)}",
        f"Rows with non-empty notes: {len(notes_df)}",
        f"Full note trail file: {outdir / 'rq1_notes_full_context.csv'}",
        "",
        "Detected issue patterns in the prose notes:",
    ]
    if tag_counts:
        summary_lines.extend([f"- {label}: {count}" for label, count in tag_counts])
    else:
        summary_lines.append("- No recurrent issue patterns detected by heuristic tagging.")

    summary_lines.extend(
        [
            "",
            "Representative examples by issue pattern:",
        ]
    )
    for label, count in tag_counts:
        summary_lines.append("")
        summary_lines.append(f"{label} ({count})")
        matching_rows = notes_df[
            notes_df["notes_issue_tags"].str.contains(label, regex=False)
        ].head(3)
        for row in matching_rows.itertuples(index=False):
            study_id = getattr(row, "study_id", "")
            title = getattr(row, "title", "")
            note_text = getattr(row, NOTES_COL, "")
            summary_lines.append(f"- Study {study_id}: {title}")
            summary_lines.append(f"  Note: {note_text}")

    (outdir / "rq1_notes_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")


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

        filtered = df_part[
            df_part["assigned_to"].astype(str).str.strip().isin([reviewer, "Ambos (piloto)"])
        ].copy()
        filtered["source_file"] = path
        frames.append(filtered)

    if not frames:
        raise ValueError("No input data was loaded.")

    return pd.concat(frames, ignore_index=True)


def build_stage_long_df(df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    carry_cols = [
        "study_id",
        "title",
        "year",
        "doi",
        "assigned_to",
        "source_file",
        "source_modality_display",
        "ai_method_display",
        "algorithm_display_raw",
        "stage_assignment_origin",
    ]

    for row in df[carry_cols + ["stage_assignments"]].to_dict("records"):
        stages = row.pop("stage_assignments")
        for stage in stages:
            new_row = dict(row)
            new_row["stage"] = stage
            records.append(new_row)

    return pd.DataFrame(records)


def cleanup_previous_outputs(outdir: Path) -> None:
    for path in outdir.glob("rq1_*"):
        if path.is_file() and path.name not in CORE_OUTPUT_FILES:
            path.unlink()


def cleanup_supporting_outputs(outdir: Path) -> None:
    for path in outdir.glob("rq1_*"):
        if path.is_file() and path.name not in SUPPORTING_OUTPUT_FILES:
            path.unlink()


# ----------------------------
# RQ1 pipeline
# ----------------------------
def main() -> None:
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    support_dir = Path(SUPPORTING_DIR)
    support_dir.mkdir(parents=True, exist_ok=True)

    df = load_reviewer_inputs()

    required_cols = [MODALITY_COL, STAGE_COL, AI_METHOD_COL, ALGORITHM_COL]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}\nAvailable columns: {list(df.columns)}"
        )

    df["modality_clean"] = unify_case_variants(df[MODALITY_COL].apply(clean_cell_value))
    df["stage_clean"] = unify_case_variants(df[STAGE_COL].apply(clean_cell_value))
    df["ai_method_clean"] = unify_case_variants(df[AI_METHOD_COL].apply(clean_cell_value))
    df["algorithm_clean"] = unify_case_variants(df[ALGORITHM_COL].apply(clean_cell_value))
    df["source_modality_display"] = translate_to_english(df["modality_clean"], MODALITY_TRANSLATIONS)
    df["ai_method_display"] = translate_to_english(df["ai_method_clean"], METHOD_TRANSLATIONS)
    df["ai_method_display"] = df["ai_method_display"].fillna("Not specified")
    df["algorithm_display_raw"] = df["algorithm_clean"].map(normalize_algorithm_value)

    stage_assignments = df.apply(extract_stage_assignments, axis=1, result_type="expand")
    df["stage_assignments"] = stage_assignments[0]
    df["stage_assignment_origin"] = stage_assignments[1]
    df["n_stage_assignments"] = df["stage_assignments"].map(len)

    analysis_mask = df["source_modality_display"].notna() & df["n_stage_assignments"].gt(0)
    df_analyzed = df[analysis_mask].copy()
    df_omitted = df[~analysis_mask].copy()

    if df_analyzed.empty:
        raise ValueError("No analyzable rows available after resolving source modality and stage.")

    stage_long_df = build_stage_long_df(df_analyzed)
    if stage_long_df.empty:
        raise ValueError("Stage expansion produced an empty dataset.")

    stage_long_df["algorithm_display"], algorithm_order = compress_algorithm_labels(
        stage_long_df["algorithm_display_raw"],
        topk=ALGORITHM_TOPK,
    )
    stage_long_df["algorithm_paper"], paper_algorithm_order = keep_all_algorithm_labels(
        stage_long_df["algorithm_display_raw"],
    )

    modality_order = ordered_categories(
        stage_long_df["source_modality_display"].dropna().astype(str).unique().tolist(),
        MODALITY_ORDER,
    )
    method_order = ordered_categories(
        stage_long_df["ai_method_display"].dropna().astype(str).unique().tolist(),
        METHOD_ORDER,
    )
    publication_flow_df = (
        stage_long_df.groupby(
            ["stage", "source_modality_display", "ai_method_display", "algorithm_paper"]
        )
        .size()
        .reset_index(name="count")
    )

    save_candidate_alluvial(
        flow_df=publication_flow_df,
        modality_order=modality_order,
        method_order=method_order,
        algorithm_order=paper_algorithm_order,
        outpath=outdir / "rq1_candidate_alluvial.png",
    )
    save_candidate_panel(
        flow_df=publication_flow_df,
        modality_order=modality_order,
        method_order=method_order,
        algorithm_order=paper_algorithm_order,
        outpath=outdir / "rq1_candidate_panel.png",
    )

    method_counts = (
        stage_long_df.groupby(["stage", "source_modality_display", "ai_method_display"])
        .size()
        .reset_index(name="count")
    )
    method_counts.to_csv(outdir / "rq1_counts_source_modality_x_method_by_stage.csv", index=False)

    method_tables_by_stage: dict[str, pd.DataFrame] = {}
    for stage in STAGE_ORDER:
        subset = method_counts[method_counts["stage"] == stage]
        table = subset.pivot(
            index="source_modality_display",
            columns="ai_method_display",
            values="count",
        ).fillna(0)
        method_tables_by_stage[stage] = table.reindex(
            index=modality_order,
            columns=method_order,
            fill_value=0,
        ).astype(int)

    save_faceted_heatmaps(
        tables_by_stage=method_tables_by_stage,
        outpath=outdir / "rq1_heatmap_source_modality_x_method_by_stage.png",
        xlabel="AI method",
        ylabel="Source modality",
        cmap="YlOrRd",
    )

    algorithm_counts = (
        stage_long_df.groupby(["stage", "algorithm_display", "source_modality_display"])
        .size()
        .reset_index(name="count")
    )
    algorithm_counts.to_csv(
        outdir / "rq1_counts_algorithm_x_source_modality_by_stage.csv",
        index=False,
    )

    algorithm_tables_by_stage: dict[str, pd.DataFrame] = {}
    for stage in STAGE_ORDER:
        subset = algorithm_counts[algorithm_counts["stage"] == stage]
        table = subset.pivot(
            index="algorithm_display",
            columns="source_modality_display",
            values="count",
        ).fillna(0)
        algorithm_tables_by_stage[stage] = table.reindex(
            index=algorithm_order,
            columns=modality_order,
            fill_value=0,
        ).astype(int)

    save_faceted_heatmaps(
        tables_by_stage=algorithm_tables_by_stage,
        outpath=outdir / "rq1_heatmap_algorithm_x_source_modality_by_stage.png",
        xlabel="Source modality",
        ylabel="Main algorithm",
        cmap="PuBuGn",
    )

    pd.DataFrame(
        [
            {"algorithm_raw": raw, "algorithm_plot": plot}
            for raw, plot in (
                stage_long_df[["algorithm_display_raw", "algorithm_display"]]
                .drop_duplicates()
                .sort_values(["algorithm_display", "algorithm_display_raw"])
                .itertuples(index=False, name=None)
            )
        ]
    ).to_csv(outdir / "rq1_algorithm_plot_mapping.csv", index=False)
    cleanup_previous_outputs(outdir)
    run_notes_audit(df_analyzed, support_dir)
    cleanup_supporting_outputs(support_dir)

    summary = Table(title="RQ1 Summary")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", justify="right", style="bold")
    summary.add_row("Total rows loaded", f"{len(df)}")
    summary.add_row("Rows in stage-resolved analysis", f"{len(df_analyzed)}")
    summary.add_row("Stage assignments after expansion", f"{len(stage_long_df)}")
    summary.add_row("Rows omitted from stage-resolved analysis", f"{len(df_omitted)}")
    summary.add_row("Resolved from explicit stage flags", f"{int((df['stage_assignment_origin'] == 'stage_flags').sum())}")
    summary.add_row("Resolved from primary-stage fallback", f"{int((df['stage_assignment_origin'] == 'stage_primary_fallback').sum())}")
    summary.add_row("Unresolved stages", f"{int((df['stage_assignment_origin'] == 'unresolved').sum())}")
    summary.add_row("Core outputs kept", f"{len(CORE_OUTPUT_FILES)}")
    console.print(summary)

    outputs_table = Table(title="RQ1 Outputs")
    outputs_table.add_column("File", style="cyan")
    for filename in sorted(CORE_OUTPUT_FILES):
        outputs_table.add_row(filename)
    console.print(outputs_table)

    supporting_table = Table(title="RQ1 Supporting Outputs")
    supporting_table.add_column("File", style="cyan")
    for filename in sorted(SUPPORTING_OUTPUT_FILES):
        supporting_table.add_row(filename)
    console.print(supporting_table)

    console.print(f"[green]RQ1 outputs saved to:[/green] {outdir.resolve()}")
    console.print(f"[green]RQ1 supporting files saved to:[/green] {support_dir.resolve()}")


if __name__ == "__main__":
    main()
