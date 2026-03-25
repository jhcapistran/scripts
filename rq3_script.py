"""
RQ3 methodological practices by AI-technique/data-source/stage combination.

This script answers:
"¿Con qué frecuencia se implementan las prácticas metodológicas clave a
través de las combinaciones de técnica de IA, fuente de datos y etapa
funcional del proceso clínico en TEA, particularmente en relación con
validación externa, integración multifuente, explicabilidad del modelo y
robustez entre sitios?"

Outputs:
- rq3_results/rq3_practice_combo_matrix.png
- rq3_results/rq3_practice_small_multiples.png
- rq3_results/rq3_practice_lollipop.png
- rq3_results/rq3_practice_by_combo.csv
- rq3_results/rq3_overall_practice_summary.csv
- rq3_results/rq3_zero_practice_combos.csv
- rq3_results/rq3_coding_coverage.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib import patches
import pandas as pd
import scienceplots


INPUT_FILES = [
    {"path": "aceptados_300_v2_codificado_RA.xlsx", "reviewer": "Revisor A"},
    {"path": "aceptados_300_codificado_RB.xlsx", "reviewer": "Revisor B"},
]
SHEET_NAME = 0
OUTPUT_DIR = "rq3_results"

METHOD_COL = "tipo_IA"
MODALITY_COL = "modalidad"
STAGE_COL = "stage_primary"

PRACTICE_COLS = {
    "q3_external_validation_signal": "External validation",
    "q3_multisource_strategy_signal": "Multisource integration",
    "q3_explainability_signal": "Model explainability",
    "q3_multisite_signal": "Cross-site robustness",
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

MODALITY_TRANSLATIONS = {
    "imagen": "Image",
    "señales fisiológicas": "Physiological signals",
    "senales fisiologicas": "Physiological signals",
    "texto · nlp": "Text / NLP",
    "texto / nlp": "Text / NLP",
    "audio · voz": "Audio / Voice",
    "audio / voz": "Audio / Voice",
    "multimodal": "Multimodal",
    "no especificado": "Not specified",
    "no especificada": "Not specified",
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
}

plt.style.use(["science", "no-latex"])


def clean_text(x: object) -> object:
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()
    return s if s else pd.NA


def normalize_category(x: object, mapping: dict[str, str], fallback: str = "Not specified") -> str:
    value = clean_text(x)
    if pd.isna(value):
        return fallback
    return mapping.get(str(value).casefold(), str(value))


def normalize_signal(x: object) -> int:
    if pd.isna(x):
        return 0
    try:
        return 1 if float(x) > 0 else 0
    except (TypeError, ValueError):
        s = str(x).strip().casefold()
        return 1 if s in {"true", "yes", "y"} else 0


def ordered_categories(observed: list[str], preferred: list[str]) -> list[str]:
    ordered = [x for x in preferred if x in observed]
    extras = sorted(x for x in observed if x not in preferred)
    return ordered + extras


def wrap_label(text: str, width: int = 22) -> str:
    words = text.split()
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


def load_inputs() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for spec in INPUT_FILES:
        df = pd.read_excel(spec["path"], sheet_name=SHEET_NAME)
        df = df[df["assigned_to"].astype(str).str.strip() == spec["reviewer"]].copy()
        df["reviewer_source"] = spec["reviewer"]
        df["source_file"] = spec["path"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_combo_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    practice_cols = list(PRACTICE_COLS.keys())
    df["method_norm"] = df[METHOD_COL].apply(normalize_category, mapping=METHOD_TRANSLATIONS)
    df["modality_norm"] = df[MODALITY_COL].apply(normalize_category, mapping=MODALITY_TRANSLATIONS)
    df["stage_norm"] = df[STAGE_COL].apply(normalize_category, mapping=STAGE_TRANSLATIONS)

    for col in practice_cols:
        df[col] = df[col].apply(normalize_signal)

    combo_cols = ["stage_norm", "modality_norm", "method_norm"]
    combo = (
        df.groupby(combo_cols, dropna=False)
        .agg(
            combo_n=("study_id", "size"),
            **{f"{col}_count": (col, "sum") for col in practice_cols},
        )
        .reset_index()
    )

    for col in practice_cols:
        combo[f"{col}_rate"] = combo[f"{col}_count"] / combo["combo_n"]

    combo["positive_practice_total"] = combo[[f"{col}_count" for col in practice_cols]].sum(axis=1)
    combo["has_any_positive"] = combo["positive_practice_total"] > 0
    combo["combo_label"] = combo.apply(
        lambda row: f"{row['stage_norm']} | {row['modality_norm']} | {row['method_norm']} (n={int(row['combo_n'])})",
        axis=1,
    )

    stage_order = ordered_categories(combo["stage_norm"].unique().tolist(), STAGE_ORDER)
    modality_order = ordered_categories(combo["modality_norm"].unique().tolist(), MODALITY_ORDER)
    method_order = ordered_categories(combo["method_norm"].unique().tolist(), METHOD_ORDER)

    combo["stage_norm"] = pd.Categorical(combo["stage_norm"], categories=stage_order, ordered=True)
    combo["modality_norm"] = pd.Categorical(combo["modality_norm"], categories=modality_order, ordered=True)
    combo["method_norm"] = pd.Categorical(combo["method_norm"], categories=method_order, ordered=True)

    combo = combo.sort_values(
        ["stage_norm", "positive_practice_total", "combo_n", "modality_norm", "method_norm"],
        ascending=[True, False, False, True, True],
    ).reset_index(drop=True)

    overall = pd.DataFrame(
        {
            "practice": [PRACTICE_COLS[col] for col in practice_cols],
            "positive_studies": [int(df[col].sum()) for col in practice_cols],
            "coded_studies": len(df),
            "positive_rate": [float(df[col].mean()) for col in practice_cols],
        }
    )
    return combo, overall


def draw_matrix(combo: pd.DataFrame, outpath: Path) -> None:
    practice_cols = list(PRACTICE_COLS.keys())
    practice_labels = list(PRACTICE_COLS.values())
    plot_df = combo[combo["has_any_positive"]].copy()

    if plot_df.empty:
        raise ValueError("No positive methodological-practice signals were found in the coded Q3 subset.")

    rates = plot_df[[f"{col}_rate" for col in practice_cols]].to_numpy()
    counts = plot_df[[f"{col}_count" for col in practice_cols]].to_numpy()
    totals = plot_df["combo_n"].to_numpy()

    n_rows = len(plot_df)
    fig_height = max(5.5, 0.46 * n_rows + 1.8)
    fig, ax = plt.subplots(figsize=(9.5, fig_height), facecolor="white")
    ax.set_facecolor("white")

    cmap = plt.cm.Blues
    norm = plt.Normalize(vmin=0, vmax=1)

    for row_idx in range(n_rows):
        for col_idx, practice in enumerate(practice_cols):
            rate = float(rates[row_idx, col_idx])
            count = int(counts[row_idx, col_idx])
            rect = patches.FancyBboxPatch(
                (col_idx, row_idx),
                1,
                1,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                linewidth=0.9,
                edgecolor="#d9d9d9",
                facecolor=cmap(norm(rate)) if count else "white",
            )
            ax.add_patch(rect)
            label = "0" if count == 0 else f"{count}/{int(totals[row_idx])}\n{rate * 100:.0f}%"
            ax.text(
                col_idx + 0.5,
                row_idx + 0.5,
                label,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if rate >= 0.55 else "#243447",
                fontweight="bold" if count else "normal",
            )

    ax.set_xlim(0, len(practice_cols))
    ax.set_ylim(n_rows, 0)
    ax.set_xticks([x + 0.5 for x in range(len(practice_cols))])
    ax.set_xticklabels([wrap_label(label, 18) for label in practice_labels], fontsize=10)
    ax.set_yticks([y + 0.5 for y in range(n_rows)])
    ax.set_yticklabels([wrap_label(label, 42) for label in plot_df["combo_label"]], fontsize=9)
    ax.tick_params(length=0)
    ax.set_xlabel("Methodological practice", fontsize=11, labelpad=10)
    ax.set_ylabel("AI technique | data source | clinical stage combination", fontsize=11, labelpad=10)

    for spine in ax.spines.values():
        spine.set_visible(False)

    stage_breaks = (
        plot_df.reset_index()
        .groupby("stage_norm", observed=False)["index"]
        .agg(["min", "max"])
        .reset_index()
    )
    for row in stage_breaks.itertuples(index=False):
        ax.hlines(row.max + 1, 0, len(practice_cols), color="#c7c7c7", linewidth=1.0)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Practice frequency within combination", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
    cbar.outline.set_visible(False)

    fig.subplots_adjust(left=0.37, right=0.93, bottom=0.1, top=0.98)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_small_multiples(combo: pd.DataFrame, outpath: Path) -> None:
    practice_cols = list(PRACTICE_COLS.keys())
    practice_labels = list(PRACTICE_COLS.values())
    panel_colors = {
        "q3_external_validation_signal": "#d55e00",
        "q3_multisource_strategy_signal": "#0072b2",
        "q3_explainability_signal": "#009e73",
        "q3_multisite_signal": "#7b6fd0",
    }

    max_total = max(int(combo["combo_n"].max()), 1)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor="white", sharex=True)
    axes = axes.flatten()

    for ax, practice_col, practice_label in zip(axes, practice_cols, practice_labels):
        plot_df = combo[combo[f"{practice_col}_count"] > 0].copy()
        ax.set_facecolor("white")

        if plot_df.empty:
            ax.text(0.5, 0.5, "No positive cases", ha="center", va="center", fontsize=11)
            ax.set_title(practice_label, fontsize=12, pad=8)
            ax.set_xlim(0, max_total + 1)
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            continue

        plot_df = plot_df.sort_values(
            [f"{practice_col}_count", f"{practice_col}_rate", "combo_n", "stage_norm"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        plot_df["label_short"] = plot_df.apply(
            lambda row: (
                f"{row['stage_norm']} | {row['modality_norm']} | {row['method_norm']}\n"
                f"(n={int(row['combo_n'])})"
            ),
            axis=1,
        )

        y_positions = list(range(len(plot_df)))
        total_vals = plot_df["combo_n"].tolist()
        positive_vals = plot_df[f"{practice_col}_count"].tolist()
        rate_vals = plot_df[f"{practice_col}_rate"].tolist()

        ax.barh(
            y_positions,
            total_vals,
            color="#d9dde3",
            edgecolor="none",
            height=0.72,
            zorder=1,
        )
        ax.barh(
            y_positions,
            positive_vals,
            color=panel_colors[practice_col],
            edgecolor="none",
            height=0.48,
            zorder=2,
        )

        for idx, (count, total, rate) in enumerate(zip(positive_vals, total_vals, rate_vals)):
            ax.text(
                min(total, max_total) + 0.18,
                idx,
                f"{int(count)}/{int(total)} ({rate * 100:.0f}%)",
                va="center",
                ha="left",
                fontsize=9,
                color="#243447",
            )

        ax.set_yticks(y_positions)
        ax.set_yticklabels([wrap_label(label, 34) for label in plot_df["label_short"]], fontsize=9)
        ax.invert_yaxis()
        ax.set_title(practice_label, fontsize=12, pad=8)
        ax.set_xlim(0, max_total + 1.6)
        ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax in axes[2:]:
        ax.set_xlabel("Studies in each combination", fontsize=10, labelpad=8)

    fig.subplots_adjust(left=0.28, right=0.97, bottom=0.08, top=0.95, wspace=0.35, hspace=0.38)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_lollipop(combo: pd.DataFrame, outpath: Path) -> None:
    practice_cols = list(PRACTICE_COLS.keys())
    plot_df = combo[combo["has_any_positive"]].copy()

    if plot_df.empty:
        raise ValueError("No positive methodological-practice signals were found in the coded Q3 subset.")

    plot_df["max_rate"] = plot_df[[f"{col}_rate" for col in practice_cols]].max(axis=1)
    plot_df = plot_df.sort_values(
        ["stage_norm", "max_rate", "positive_practice_total", "combo_n"],
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    practice_styles = {
        "q3_external_validation_signal": {"color": "#d55e00", "marker": "o"},
        "q3_multisource_strategy_signal": {"color": "#0072b2", "marker": "s"},
        "q3_explainability_signal": {"color": "#009e73", "marker": "D"},
        "q3_multisite_signal": {"color": "#7b6fd0", "marker": "^"},
    }
    offsets = {
        "q3_external_validation_signal": -0.24,
        "q3_multisource_strategy_signal": -0.08,
        "q3_explainability_signal": 0.08,
        "q3_multisite_signal": 0.24,
    }

    n_rows = len(plot_df)
    fig_height = max(5.5, 0.52 * n_rows + 1.6)
    fig, ax = plt.subplots(figsize=(11.2, fig_height), facecolor="white")
    ax.set_facecolor("white")

    for base_y in range(n_rows):
        ax.hlines(base_y, 0, 100, color="#eceff3", linewidth=0.8, zorder=0)

    for row_idx, row in plot_df.iterrows():
        total = int(row["combo_n"])
        for practice_col in practice_cols:
            count = int(row[f"{practice_col}_count"])
            if count <= 0:
                continue
            rate = float(row[f"{practice_col}_rate"]) * 100
            y = row_idx + offsets[practice_col]
            style = practice_styles[practice_col]
            ax.hlines(y, 0, rate, color=style["color"], linewidth=2.2, alpha=0.9, zorder=2)
            ax.scatter(
                [rate],
                [y],
                s=64,
                color=style["color"],
                marker=style["marker"],
                edgecolors="white",
                linewidths=0.8,
                zorder=3,
            )
            ax.text(
                min(rate + 2.0, 102.5),
                y,
                f"{count}/{total}",
                va="center",
                ha="left",
                fontsize=8.5,
                color="#243447",
            )

    ax.set_xlim(0, 104)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([wrap_label(label, 42) for label in plot_df["combo_label"]], fontsize=9)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Implementation frequency within each combination", fontsize=11, labelpad=10)
    ax.set_ylabel("AI technique | data source | clinical stage combination", fontsize=11, labelpad=10)
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.9)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    stage_breaks = (
        plot_df.reset_index()
        .groupby("stage_norm", observed=False)["index"]
        .agg(["min", "max"])
        .reset_index()
    )
    for row in stage_breaks.itertuples(index=False):
        ax.hlines(row.max + 0.5, 0, 100, color="#c7c7c7", linewidth=1.0)

    legend_handles = []
    for practice_col in practice_cols:
        style = practice_styles[practice_col]
        handle = plt.Line2D(
            [0],
            [0],
            color=style["color"],
            marker=style["marker"],
            linestyle="-",
            linewidth=2.0,
            markersize=6,
            label=PRACTICE_COLS[practice_col],
        )
        legend_handles.append(handle)
    ax.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=9,
        loc="lower left",
        bbox_to_anchor=(0.0, -0.16),
        ncol=2,
    )

    fig.subplots_adjust(left=0.4, right=0.97, bottom=0.12, top=0.98)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_inputs()
    practice_cols = list(PRACTICE_COLS.keys())

    coverage = (
        df.groupby("reviewer_source")[practice_cols]
        .apply(lambda x: (~x.isna().all(axis=1)).sum())
        .reset_index(name="q3_coded_rows")
    )
    coverage["total_rows"] = df.groupby("reviewer_source").size().values
    coverage["q3_coded_rate"] = coverage["q3_coded_rows"] / coverage["total_rows"]
    coverage.to_csv(outdir / "rq3_coding_coverage.csv", index=False)

    coded_df = df[~df[practice_cols].isna().all(axis=1)].copy()
    if coded_df.empty:
        raise ValueError("No Q3-coded rows were found. All practice signals are missing.")

    combo, overall = build_combo_summary(coded_df)
    combo.to_csv(outdir / "rq3_practice_by_combo.csv", index=False)
    overall.to_csv(outdir / "rq3_overall_practice_summary.csv", index=False)
    combo.loc[~combo["has_any_positive"]].to_csv(outdir / "rq3_zero_practice_combos.csv", index=False)

    draw_matrix(combo, outdir / "rq3_practice_combo_matrix.png")
    draw_small_multiples(combo, outdir / "rq3_practice_small_multiples.png")
    draw_lollipop(combo, outdir / "rq3_practice_lollipop.png")

    print(f"RQ3 outputs saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
