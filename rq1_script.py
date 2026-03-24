"""
RQ1 plots generator (no argparse).

This version keeps the legacy overview outputs and adds new stage-faceted
heatmaps that directly answer the RQ1 wording about the distribution of
AI methods and algorithms across source modalities and clinical stages.

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

MODALITY_COL = "modalidad"
STAGE_COL = "stage_primary"
AI_METHOD_COL = "tipo_IA"
ALGORITHM_COL = "AI_algorithm_main"

STAGE_FLAG_COLS = {
    "stage_prescreening": "Prescreening",
    "stage_screening": "Screening",
    "stage_diagnosis": "Diagnosis",
    "stage_prognosis": "Prognosis",
    "stage_monitoring_intervention": "Monitoring/intervention",
}

AUDIT_NOTES = True
NOTES_COL = "notes_coding"
NOTES_TOPK = 40
ALGORITHM_TOPK = 8
# ----------------------------


STAGE_ORDER = [
    "Prescreening",
    "Screening",
    "Diagnosis",
    "Prognosis",
    "Monitoring/intervention",
]
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


def save_heatmap(
    table: pd.DataFrame,
    outpath: Path,
    xlabel: str,
    ylabel: str,
    title: str,
    cmap: str = "Blues",
) -> None:
    fig_w = max(8, 1.2 * table.shape[1] + 4)
    fig_h = max(6, 0.65 * table.shape[0] + 3)
    vmax = int(table.to_numpy().max()) if table.size else 1

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = draw_heatmap(
        ax=ax,
        table=table,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        cmap=cmap,
        vmax=vmax,
    )
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


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


def save_bar(series: pd.Series, outpath: Path, xlabel: str, ylabel: str = "Count") -> None:
    fig_w = max(8, 0.7 * len(series) + 4)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    ax.bar(series.index.astype(str), series.values)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


# ----------------------------
# Notes audit helpers
# ----------------------------
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by", "from",
    "as", "at", "is", "are", "was", "were", "be", "been", "being", "this", "that", "these",
    "those", "it", "its", "we", "they", "their", "our", "not", "no", "yes", "na", "n/a",
    "el", "la", "los", "las", "un", "una", "unos", "unas", "y", "o", "de", "del", "al",
    "para", "con", "por", "en", "es", "son", "fue", "eran", "ser", "como", "este", "esta",
    "estos", "estas", "su", "sus", "no", "si",
}
TOKEN_RE = re.compile(r"[a-zA-Z0-9]+(?:'[a-zA-Z0-9]+)?")
console = Console()


def tokenize(text: str) -> list[str]:
    tokens = [t.lower() for t in TOKEN_RE.findall(text)]
    return [t for t in tokens if len(t) >= 3 and t not in STOPWORDS]


def top_bigrams(tokens: list[str]) -> list[tuple[str, str]]:
    return list(zip(tokens, tokens[1:])) if len(tokens) >= 2 else []


def run_notes_audit(df: pd.DataFrame, outdir: Path) -> None:
    if NOTES_COL not in df.columns:
        return

    notes = df[NOTES_COL].dropna().astype(str)
    notes = notes[notes.str.strip().astype(bool)]

    all_tokens: list[str] = []
    all_bigrams: list[tuple[str, str]] = []

    for s in notes:
        toks = tokenize(s)
        all_tokens.extend(toks)
        all_bigrams.extend(top_bigrams(toks))

    tok_counts = Counter(all_tokens).most_common(NOTES_TOPK)
    bigram_counts = Counter(all_bigrams).most_common(NOTES_TOPK)

    pd.DataFrame(tok_counts, columns=["token", "count"]).to_csv(
        outdir / "rq1_notes_top_tokens.csv", index=False
    )
    pd.DataFrame(
        [(f"{a} {b}", c) for (a, b), c in bigram_counts],
        columns=["bigram", "count"],
    ).to_csv(outdir / "rq1_notes_top_bigrams.csv", index=False)

    summary_lines = [
        f"Notes column: {NOTES_COL}",
        f"Total analyzed rows: {len(df)}",
        f"Non-empty notes: {len(notes)}",
        f"Unique tokens: {len(set(all_tokens))}",
        f"Unique bigrams: {len(set(all_bigrams))}",
        "",
        f"Top {min(NOTES_TOPK, len(tok_counts))} tokens (token,count):",
        *[f"{t},{c}" for t, c in tok_counts],
        "",
        f"Top {min(NOTES_TOPK, len(bigram_counts))} bigrams (bigram,count):",
        *[f"{a} {b},{c}" for (a, b), c in bigram_counts],
    ]
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
    if NOTES_COL in df.columns:
        carry_cols.append(NOTES_COL)

    for row in df[carry_cols + ["stage_assignments"]].to_dict("records"):
        stages = row.pop("stage_assignments")
        for stage in stages:
            new_row = dict(row)
            new_row["stage"] = stage
            records.append(new_row)

    return pd.DataFrame(records)


def crosstab_with_order(
    index_series: pd.Series,
    column_series: pd.Series,
    index_order: list[str],
    column_order: list[str],
) -> pd.DataFrame:
    table = pd.crosstab(index_series, column_series)
    return table.reindex(index=index_order, columns=column_order, fill_value=0).astype(int)


# ----------------------------
# RQ1 pipeline
# ----------------------------
def main() -> None:
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

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
    df["stage_display_primary"] = df["stage_clean"].map(normalize_stage_value)
    df["ai_method_display"] = translate_to_english(df["ai_method_clean"], METHOD_TRANSLATIONS)
    df["ai_method_display"] = df["ai_method_display"].fillna("Not specified")
    df["algorithm_display_raw"] = df["algorithm_clean"].map(normalize_algorithm_value)
    df["modality_display"] = translate_to_english(
        df["modality_clean"],
        {
            "imagen": "Image",
            "señales fisiológicas": "Physiological signals",
            "senales fisiologicas": "Physiological signals",
            "texto · nlp": "Text / NLP",
            "texto / nlp": "Text / NLP",
            "audio · voz": "Audio / Voice",
            "audio / voz": "Audio / Voice",
            "no especificado": "Not specified",
            "no especificada": "Not specified",
        },
    )
    df["stage_display"] = translate_to_english(
        df["stage_clean"],
        {
            "monitoring_intervention": "Monitoring/intervention",
            "monitoring/intervention": "Monitoring/intervention",
            "diagnosis": "Diagnosis",
            "screening": "Screening",
            "prognosis": "Prognosis",
            "prescreening": "Prescreening",
            "not clear": "Not clear",
        },
    )

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
    df_analyzed["algorithm_display"] = df_analyzed["algorithm_display_raw"].map(
        lambda x: x if x in algorithm_order or x == UNCLEAR_ALGORITHM_LABEL else OTHER_ALGORITHM_LABEL
    )

    modality_order = ordered_categories(
        stage_long_df["source_modality_display"].dropna().astype(str).unique().tolist(),
        MODALITY_ORDER,
    )
    method_order = ordered_categories(
        stage_long_df["ai_method_display"].dropna().astype(str).unique().tolist(),
        METHOD_ORDER,
    )
    primary_stage_order = ordered_categories(
        df["stage_display_primary"].dropna().astype(str).unique().tolist(),
        STAGE_ORDER + ["Not clear"],
    )

    legacy_mask = df["source_modality_display"].notna() & df["stage_display_primary"].notna()
    df_legacy = df[legacy_mask].copy()
    legacy_table = crosstab_with_order(
        df_legacy["source_modality_display"],
        df_legacy["stage_display_primary"],
        modality_order,
        primary_stage_order,
    )
    legacy_table.to_csv(outdir / "rq1_table_modality_x_stage.csv", index=True)
    save_heatmap(
        table=legacy_table,
        outpath=outdir / "rq1_heatmap_modality_x_stage.png",
        xlabel="Clinical functional stage (primary coding)",
        ylabel="Source modality",
        title="RQ1 overview: source modality x primary clinical stage",
        cmap="Blues",
    )

    stage_dist = df_legacy["stage_display_primary"].value_counts().reindex(primary_stage_order, fill_value=0)
    modality_dist = df_legacy["source_modality_display"].value_counts().reindex(modality_order, fill_value=0)

    save_bar(
        stage_dist,
        outdir / "rq1_stage_distribution.png",
        xlabel="Clinical functional stage (primary coding)",
    )
    save_bar(
        modality_dist,
        outdir / "rq1_modality_distribution.png",
        xlabel="Source modality",
    )

    stage_assignment_table = crosstab_with_order(
        stage_long_df["source_modality_display"],
        stage_long_df["stage"],
        modality_order,
        STAGE_ORDER,
    )
    stage_assignment_table.to_csv(
        outdir / "rq1_table_source_modality_x_stage_assignments.csv",
        index=True,
    )
    save_heatmap(
        table=stage_assignment_table,
        outpath=outdir / "rq1_heatmap_source_modality_x_stage_assignments.png",
        xlabel="Clinical functional stage (all coded assignments)",
        ylabel="Source modality",
        title="RQ1 stage-resolved overview: source modality x clinical stage",
        cmap="YlGnBu",
    )

    method_source_table = crosstab_with_order(
        df_analyzed["source_modality_display"],
        df_analyzed["ai_method_display"],
        modality_order,
        method_order,
    )
    method_stage_table = crosstab_with_order(
        stage_long_df["ai_method_display"],
        stage_long_df["stage"],
        method_order,
        STAGE_ORDER,
    )
    algorithm_source_table = crosstab_with_order(
        df_analyzed["algorithm_display"],
        df_analyzed["source_modality_display"],
        algorithm_order,
        modality_order,
    )
    algorithm_stage_table = crosstab_with_order(
        stage_long_df["algorithm_display"],
        stage_long_df["stage"],
        algorithm_order,
        STAGE_ORDER,
    )

    method_source_table.to_csv(outdir / "rq1_table_method_x_source_modality.csv", index=True)
    method_stage_table.to_csv(outdir / "rq1_table_method_x_stage.csv", index=True)
    algorithm_source_table.to_csv(outdir / "rq1_table_algorithm_x_source_modality.csv", index=True)
    algorithm_stage_table.to_csv(outdir / "rq1_table_algorithm_x_stage.csv", index=True)

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

    df_analyzed.to_csv(outdir / "rq1_review_rows_analyzed.csv", index=False)
    stage_long_df.to_csv(outdir / "rq1_stage_assignments_long.csv", index=False)
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

    report_df = pd.DataFrame(
        [
            {
                "field": MODALITY_COL,
                "n_total": len(df),
                "n_empty_raw": int(df["modality_clean"].isna().sum()),
                "n_resolved_for_plot": int(df["source_modality_display"].notna().sum()),
                "n_omitted_from_stage_analysis": int(df_omitted.shape[0]),
            },
            {
                "field": STAGE_COL,
                "n_total": len(df),
                "n_empty_raw": int(df["stage_clean"].isna().sum()),
                "n_resolved_for_plot": int(df["n_stage_assignments"].gt(0).sum()),
                "n_omitted_from_stage_analysis": int(df["n_stage_assignments"].eq(0).sum()),
            },
            {
                "field": AI_METHOD_COL,
                "n_total": len(df),
                "n_empty_raw": int(df["ai_method_clean"].isna().sum()),
                "n_resolved_for_plot": int(df["ai_method_display"].notna().sum()),
                "n_omitted_from_stage_analysis": int(df_omitted.shape[0]),
            },
            {
                "field": ALGORITHM_COL,
                "n_total": len(df),
                "n_empty_raw": int(df["algorithm_clean"].isna().sum()),
                "n_resolved_for_plot": int(df["algorithm_display_raw"].notna().sum()),
                "n_omitted_from_stage_analysis": int(df_omitted.shape[0]),
            },
        ]
    )
    report_df.to_csv(outdir / "rq1_missingness_report.csv", index=False)

    preferred_cols = [
        "study_id",
        "title",
        "year",
        "doi",
        "assigned_to",
        MODALITY_COL,
        "source_modality_display",
        STAGE_COL,
        "stage_display_primary",
        "stage_assignment_origin",
        "n_stage_assignments",
        AI_METHOD_COL,
        "ai_method_display",
        ALGORITHM_COL,
        "algorithm_display_raw",
    ]
    if NOTES_COL in df_omitted.columns:
        preferred_cols.append(NOTES_COL)
    preferred_cols = [c for c in preferred_cols if c in df_omitted.columns]
    omitted_rows = df_omitted[preferred_cols] if preferred_cols else df_omitted
    omitted_rows.to_csv(outdir / "rq1_rows_omitted.csv", index=False)

    if AUDIT_NOTES:
        run_notes_audit(df_analyzed, outdir)

    notes_non_empty = 0
    if NOTES_COL in df_analyzed.columns:
        notes_non_empty = int(df_analyzed[NOTES_COL].dropna().astype(str).str.strip().astype(bool).sum())

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
    if NOTES_COL in df_analyzed.columns:
        summary.add_row("Non-empty notes in analyzed rows", f"{notes_non_empty}")
    summary.add_row("Input files combined", f"{len(INPUT_FILES)}")
    console.print(summary)

    stage_origin_summary = Table(title="Stage Resolution Summary")
    stage_origin_summary.add_column("Resolution", style="cyan")
    stage_origin_summary.add_column("Rows", justify="right", style="bold")
    for origin, count in (
        df["stage_assignment_origin"].value_counts().reindex(
            ["stage_flags", "stage_primary_fallback", "unresolved"],
            fill_value=0,
        ).items()
    ):
        stage_origin_summary.add_row(origin, str(int(count)))
    console.print(stage_origin_summary)

    if "assigned_to" in df.columns:
        reviewer_summary = Table(title="Reviewer Assignment Summary")
        reviewer_summary.add_column("Reviewer", style="cyan")
        reviewer_summary.add_column("Assigned", justify="right", style="bold")
        reviewer_summary.add_column("Analyzed", justify="right", style="bold green")
        reviewer_summary.add_column("Omitted", justify="right", style="bold red")

        reviewer_df = df.assign(analyzed=analysis_mask, omitted=~analysis_mask)
        reviewer_counts = (
            reviewer_df.groupby("assigned_to", dropna=False)
            .agg(
                assigned=("assigned_to", "size"),
                analyzed=("analyzed", "sum"),
                omitted=("omitted", "sum"),
            )
            .reset_index()
        )

        for row in reviewer_counts.itertuples(index=False):
            reviewer_name = row.assigned_to if pd.notna(row.assigned_to) else "Missing reviewer"
            reviewer_summary.add_row(
                str(reviewer_name),
                str(int(row.assigned)),
                str(int(row.analyzed)),
                str(int(row.omitted)),
            )

        console.print(reviewer_summary)

    console.print(f"[green]RQ1 outputs saved to:[/green] {outdir.resolve()}")


if __name__ == "__main__":
    main()
