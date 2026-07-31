from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import scienceplots


matplotlib.use("Agg")
plt.style.use(["science", "no-latex"])

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "RQ3_datos.xlsx"
SHEET_NAME = "RQ3_graph_ready"
OUTPUT_DIR = BASE_DIR / "rq3_results_q1_v2"
SUPPLEMENT_DIR = OUTPUT_DIR / "supplement"
MASTER_DENOMINATOR_FILE = BASE_DIR / "rq_denominators_q1_v2.csv"
# Confirmed non-primary records (reviews/surveys/perspectives) shared across RQ1/RQ2/RQ3, see rq_eligibility_recheck_q1_v2.csv.
EXCLUDED_STUDIES_FILE = BASE_DIR / "rq_excluded_studies_q1_v2.csv"

METHOD_COL = "tipo_IA"
MODALITY_COL = "modalidad"
STAGE_COL = "stage_primary"

PRACTICE_COLS = {
    "q3_external_validation_signal": ("External validation", "#d95f02", "X"),
    "q3_multisource_strategy_signal": ("Multisource integration", "#1f78b4", "s"),
    "q3_explainability_signal": ("Model explainability", "#1b9e77", "D"),
    "q3_multisite_signal": ("Cross-site robustness", "#7570b3", "^"),
}
STAGE_ORDER = [
    "Prescreening",
    "Screening",
    "Diagnosis",
    "Prognosis",
    "Monitoring/intervention",
    "Not specified",
]
METHOD_ORDER = ["Machine Learning", "Deep Learning", "Hybrid", "Not specified"]
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
    "seã±ales fisiolã³gicas": "Physiological signals",
    "texto · nlp": "Text / NLP",
    "texto / nlp": "Text / NLP",
    "audio · voz": "Audio / Voice",
    "audio / voz": "Audio / Voice",
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

# ponytail: no supplement, so the main figure carries every positive combination.
MIN_COMBO_N_FOR_FIGURE = 1
TOP_COMBOS_FOR_FIGURE = 999


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


def normalize_signal(value: object) -> int:
    if pd.isna(value):
        return 0
    try:
        return 1 if float(value) > 0 else 0
    except (TypeError, ValueError):
        return 1 if str(value).strip().casefold() in {"true", "yes", "y"} else 0


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


def load_excluded_study_ids() -> pd.DataFrame:
    if not EXCLUDED_STUDIES_FILE.exists():
        return pd.DataFrame(columns=["study_id", "tier", "title", "reason"])
    return pd.read_csv(EXCLUDED_STUDIES_FILE)


def load_base_df() -> pd.DataFrame:
    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME).copy()
    excluded = load_excluded_study_ids()
    return df[~df["study_id"].isin(excluded["study_id"])].copy()


def build_combo_summary(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["method_norm"] = work[METHOD_COL].apply(normalize_category, mapping=METHOD_TRANSLATIONS)
    work["modality_norm"] = work[MODALITY_COL].apply(normalize_category, mapping=MODALITY_TRANSLATIONS)
    work["stage_norm"] = work[STAGE_COL].apply(normalize_category, mapping=STAGE_TRANSLATIONS)

    for col in PRACTICE_COLS:
        work[col] = work[col].apply(normalize_signal)

    combo = (
        work.groupby(["stage_norm", "modality_norm", "method_norm"], dropna=False)
        .agg(
            combo_n=("study_id", "size"),
            **{f"{col}_count": (col, "sum") for col in PRACTICE_COLS},
        )
        .reset_index()
    )

    for col in PRACTICE_COLS:
        combo[f"{col}_rate"] = combo[f"{col}_count"] / combo["combo_n"]

    combo["positive_practice_total"] = combo[[f"{col}_count" for col in PRACTICE_COLS]].sum(axis=1)
    combo["max_practice_rate"] = combo[[f"{col}_rate" for col in PRACTICE_COLS]].max(axis=1)
    combo["combo_label"] = combo.apply(
        lambda row: f"{row['stage_norm']} | {row['modality_norm']} | {row['method_norm']} (n={int(row['combo_n'])})",
        axis=1,
    )

    combo["stage_norm"] = pd.Categorical(
        combo["stage_norm"],
        categories=ordered_categories(combo["stage_norm"].astype(str).unique().tolist(), STAGE_ORDER),
        ordered=True,
    )
    combo["modality_norm"] = pd.Categorical(
        combo["modality_norm"],
        categories=ordered_categories(combo["modality_norm"].astype(str).unique().tolist(), MODALITY_ORDER),
        ordered=True,
    )
    combo["method_norm"] = pd.Categorical(
        combo["method_norm"],
        categories=ordered_categories(combo["method_norm"].astype(str).unique().tolist(), METHOD_ORDER),
        ordered=True,
    )
    return combo.sort_values(
        ["combo_n", "max_practice_rate", "positive_practice_total", "stage_norm", "modality_norm", "method_norm"],
        ascending=[False, False, False, True, True, True],
    ).reset_index(drop=True)


def build_plot_summary(combo: pd.DataFrame) -> pd.DataFrame:
    work = combo.copy()
    fully_specified = (
        ~work["stage_norm"].eq("Not specified")
        & ~work["modality_norm"].eq("Not specified")
        & ~work["method_norm"].eq("Not specified")
    )
    work["plot_stage"] = work["stage_norm"].astype(str)
    work["plot_modality"] = work["modality_norm"].astype(str)
    work["plot_method"] = work["method_norm"].astype(str)
    work["specification_group"] = "Fully specified profile"

    modality_ns = ~fully_specified & work["modality_norm"].eq("Not specified")
    method_ns = ~fully_specified & work["method_norm"].eq("Not specified")
    stage_ns = ~fully_specified & work["stage_norm"].eq("Not specified")

    work.loc[modality_ns, "plot_modality"] = "Data source not specified"
    work.loc[modality_ns, "plot_method"] = "Any AI technique"
    work.loc[method_ns, "plot_modality"] = "Any data source"
    work.loc[method_ns, "plot_method"] = "AI technique not specified"
    work.loc[stage_ns, "plot_stage"] = "Clinical stage not specified"
    work.loc[stage_ns, "plot_modality"] = "Any data source"
    work.loc[stage_ns, "plot_method"] = "Any AI technique"

    work.loc[modality_ns, "specification_group"] = "Profiles with unspecified data source"
    work.loc[method_ns, "specification_group"] = "Profiles with unspecified AI technique"
    work.loc[stage_ns, "specification_group"] = "Profiles with unspecified clinical stage"

    group_cols = ["plot_stage", "plot_modality", "plot_method", "specification_group"]
    agg_map = {"combo_n": "sum", "positive_practice_total": "sum"}
    for col in PRACTICE_COLS:
        agg_map[f"{col}_count"] = "sum"
    plot = work.groupby(group_cols, dropna=False).agg(agg_map).reset_index()
    for col in PRACTICE_COLS:
        plot[f"{col}_rate"] = plot[f"{col}_count"] / plot["combo_n"]
    plot["max_practice_rate"] = plot[[f"{col}_rate" for col in PRACTICE_COLS]].max(axis=1)
    plot["plot_label"] = plot.apply(
        lambda row: (
            f"{row['plot_stage']} | {row['plot_modality']} (n={int(row['combo_n'])})"
            if row["specification_group"] == "Profiles with unspecified data source"
            else (
                f"{row['plot_stage']} | {row['plot_method']} (n={int(row['combo_n'])})"
                if row["specification_group"] == "Profiles with unspecified AI technique"
                else (
                    f"{row['plot_stage']} | {row['plot_modality']} | {row['plot_method']} (n={int(row['combo_n'])})"
                    if row["specification_group"] == "Fully specified profile"
                    else f"{row['plot_stage']} (n={int(row['combo_n'])})"
                )
            )
        ),
        axis=1,
    )
    plot["stage_sort"] = pd.Categorical(
        plot["plot_stage"],
        categories=ordered_categories(plot["plot_stage"].astype(str).unique().tolist(), STAGE_ORDER + ["Clinical stage not specified"]),
        ordered=True,
    )
    return plot.sort_values(
        ["stage_sort", "specification_group", "max_practice_rate", "combo_n", "positive_practice_total"],
        ascending=[True, True, False, False, False],
    ).reset_index(drop=True)


def build_global_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total_n = int(len(df))
    for col, (label, _, _) in PRACTICE_COLS.items():
        positive_n = int(df[col].apply(normalize_signal).sum())
        rows.append(
            {
                "practice_signal": col,
                "practice": label,
                "positive_n": positive_n,
                "total_n": total_n,
                "positive_rate": positive_n / total_n if total_n else 0.0,
            }
        )
    return pd.DataFrame(rows)


def select_plot_combos(combo: pd.DataFrame) -> pd.DataFrame:
    plot_df = build_plot_summary(combo)
    plot_df = plot_df[(plot_df["combo_n"] >= MIN_COMBO_N_FOR_FIGURE) & (plot_df["positive_practice_total"] > 0)].copy()
    if plot_df.empty:
        plot_df = build_plot_summary(combo[combo["positive_practice_total"] > 0].copy())
    return plot_df.head(TOP_COMBOS_FOR_FIGURE).sort_values(
        ["stage_sort", "specification_group", "max_practice_rate", "combo_n", "positive_practice_total"],
        ascending=[True, False, False, False, False],
    ).reset_index(drop=True)


def build_omitted_zero_profiles(combo: pd.DataFrame) -> pd.DataFrame:
    return combo[combo["positive_practice_total"] == 0].copy().reset_index(drop=True)


def draw_compact_dotplot(
    combo: pd.DataFrame,
    outpath: Path,
    panel_specs: list[tuple[str, list[str]]],
    layout: dict[str, float] | None = None,
) -> None:
    plot_df = select_plot_combos(combo)
    if plot_df.empty:
        raise ValueError("No positive Q3 combinations available to plot.")
    layout = layout or {}

    panel_data = [(title, plot_df[plot_df["plot_stage"].isin(stages)].reset_index(drop=True)) for title, stages in panel_specs]
    panel_data = [(title, df_stage) for title, df_stage in panel_data if not df_stage.empty]

    max_rows = max(len(df_stage) for _, df_stage in panel_data)
    total_rows = sum(len(df_stage) for _, df_stage in panel_data)
    n_panels = len(panel_data)
    ncols = 1
    nrows = (n_panels + ncols - 1) // ncols
    row_height = layout.get("row_height", 0.5)
    base_height = layout.get("base_height", 1.05)
    min_height = layout.get("min_height", 5.9)
    fig_height = max(min_height, row_height * total_rows + base_height * nrows)
    max_label_len = max(len(str(label)) for _, df_stage in panel_data for label in df_stage["plot_label"])
    fig_width = min(12.5, max(9.6, 7.2 + 0.055 * max_label_len))
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), facecolor="white")
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for ax, (title, stage_df) in zip(axes, panel_data):
        y_positions = list(range(len(stage_df)))
        for row_idx, row in stage_df.iterrows():
            nonzero_points = []
            for col, (_, color, marker) in PRACTICE_COLS.items():
                count = int(row[f"{col}_count"])
                if count == 0:
                    continue
                nonzero_points.append((col, color, marker, count, float(row[f"{col}_rate"])))

            groups: dict[float, list[tuple[str, str, str, int, float]]] = {}
            for point in nonzero_points:
                groups.setdefault(round(point[4], 6), []).append(point)

            row_labels: list[str] = []

            for points in groups.values():
                offsets = [0.0] if len(points) == 1 else [0.014 * (idx - (len(points) - 1) / 2) for idx in range(len(points))]
                x_offsets = [0.0] if len(points) == 1 else [0.008 * (idx - (len(points) - 1) / 2) for idx in range(len(points))]
                for (col, color, marker, count, rate), y_offset, x_offset in zip(points, offsets, x_offsets):
                    y_plot = row_idx + y_offset
                    x_plot = rate + x_offset
                    ax.scatter(x_plot, y_plot, color=color, marker=marker, s=120, edgecolors="white", linewidths=1.0, zorder=3)
                row_labels.append(f"{points[0][3]}/{int(row['combo_n'])}")

            if row_labels:
                ax.text(1.045, row_idx, ", ".join(row_labels), va="center", ha="left", fontsize=12)

        ax.set_title(title, fontsize=16, loc="left")
        ax.set_xlim(0, 1.16)
        ax.set_ylim(-0.5, len(stage_df) - 0.5)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(stage_df["plot_label"], fontsize=12)
        ax.set_ylabel("Stage | data source | AI technique", fontsize=15, labelpad=34)
        ax.invert_yaxis()
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
        ax.tick_params(axis="x", labelsize=13)
        ax.set_xlabel("Implementation frequency within each profile", fontsize=15, labelpad=12)
        ax.grid(axis="x", color="#e5e7eb", linewidth=0.9)
        ax.grid(axis="y", color="#f1f5f9", linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax in axes[len(panel_data):]:
        ax.axis("off")

    visible_practices = [
        (col, label, color, marker)
        for col, (label, color, marker) in PRACTICE_COLS.items()
        if any(int(row[f"{col}_count"]) > 0 for _, df_stage in panel_data for _, row in df_stage.iterrows())
    ]
    legend_handles = [
        Line2D([0], [0], color=color, marker=marker, linewidth=0, markersize=10, label=label)
        for _, label, color, marker in visible_practices
    ]
    legend_y = layout.get("legend_y", 0.01 if n_panels == 1 else 0.02)
    xlabel_y = layout.get("xlabel_y", 0.12 if n_panels == 1 else 0.08)
    bottom_margin = layout.get("bottom_margin", 0.2 if n_panels == 1 else 0.14)
    legend_cols = 2 if len(legend_handles) > 2 else max(1, len(legend_handles))
    fig.legend(handles=legend_handles, frameon=False, ncol=legend_cols, loc="lower center", bbox_to_anchor=(0.5, legend_y), fontsize=13)
    fig.subplots_adjust(left=0.42, right=0.98, top=0.95, bottom=bottom_margin, hspace=layout.get("hspace", 0.12))
    save_figure_variants(fig, outpath)


def refresh_master_denominator_table() -> None:
    frames = []
    for filename in ("rq1_denominators_q1_v2.csv", "rq2_denominators_q1_v2.csv", "rq3_denominators_q1_v2.csv"):
        path = BASE_DIR / ("rq1_results_q1_v2" if filename.startswith("rq1") else "rq2_results_q1_v2" if filename.startswith("rq2") else "rq3_results_q1_v2") / filename
        if path.exists():
            frames.append(pd.read_csv(path))
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(MASTER_DENOMINATOR_FILE, index=False)


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    ensure_dir(SUPPLEMENT_DIR)

    df = load_base_df()
    combo = build_combo_summary(df)
    global_summary = build_global_summary(df)

    plot_combos = select_plot_combos(combo)
    omitted_zero = build_omitted_zero_profiles(combo)

    combo.to_csv(SUPPLEMENT_DIR / "rq3_combo_summary_q1_v2.csv", index=False)
    plot_combos.to_csv(OUTPUT_DIR / "rq3_lollipop_combos_q1_v2.csv", index=False)
    omitted_zero.to_csv(OUTPUT_DIR / "rq3_omitted_zero_profiles_q1_v2.csv", index=False)
    global_summary.to_csv(OUTPUT_DIR / "rq3_global_practice_summary_q1_v2.csv", index=False)

    denominator_rows = [
        {
            "rq": "RQ3",
            "subset": "all_rows",
            "group": "All studies",
            "denominator_n": int(len(df)),
            "notes": "All studies in the consolidated sheet, after excluding confirmed non-primary records.",
        },
        {
            "rq": "RQ3",
            "subset": "exclusions_applied",
            "group": "Tier1 non-primary (reviews/surveys/perspectives)",
            "denominator_n": len(load_excluded_study_ids()),
            "notes": "Excluded before analysis; see rq_excluded_studies_q1_v2.csv.",
        },
    ]
    for _, row in global_summary.iterrows():
        denominator_rows.append(
            {
                "rq": "RQ3",
                "subset": "practice",
                "group": row["practice"],
                "denominator_n": int(row["total_n"]),
                "notes": f"Positive={int(row['positive_n'])}; rate={row['positive_rate']:.1%}.",
            }
        )
    pd.DataFrame(denominator_rows).to_csv(OUTPUT_DIR / "rq3_denominators_q1_v2.csv", index=False)

    plotted_profiles_n = int(len(plot_combos))
    plotted_studies_n = int(plot_combos["combo_n"].sum()) if not plot_combos.empty else 0
    omitted_profiles_n = int(len(omitted_zero))
    omitted_studies_n = int(omitted_zero["combo_n"].sum()) if not omitted_zero.empty else 0

    figure_specs = [
        ("rq3_practice_lollipop_a_q1_v2", "A. Prescreening and screening", ["Prescreening", "Screening"], {"row_height": 0.47, "base_height": 1.0, "min_height": 5.8}),
        ("rq3_practice_lollipop_b_q1_v2", "B. Diagnosis", ["Diagnosis"], {"row_height": 0.48, "base_height": 1.02, "min_height": 6.0}),
        ("rq3_practice_lollipop_c_q1_v2", "C. Monitoring/intervention", ["Monitoring/intervention"], {"row_height": 0.46, "base_height": 1.0, "min_height": 5.5}),
        ("rq3_practice_lollipop_d_q1_v2", "D. Prognosis and unspecified stage", ["Prognosis", "Clinical stage not specified"], {"row_height": 0.34, "base_height": 0.8, "min_height": 4.1, "bottom_margin": 0.27, "xlabel_y": 0.105, "legend_y": 0.01}),
    ]
    for stem, title, stages, layout in figure_specs:
        draw_compact_dotplot(combo, OUTPUT_DIR / stem, [(title, stages)], layout)
    write_caption(
        OUTPUT_DIR / "rq3_practice_lollipop_a_q1_v2_caption.txt",
        f"""
        RQ3A (prescreening and screening). Compact dot plot showing how frequently four methodological practices are implemented within combinations of
        clinical stage, data source, and AI technique in ASD studies. Each row represents one fully specified profile
        or one aggregated partially specified profile with at least one positive methodological-practice signal in the
        consolidated dataset. Profiles with unspecified data source, AI technique, or clinical stage were collapsed into
        compact classes for readability, while preserving their counts in the plotted n/N labels. Marker position
        encodes within-profile frequency, and adjacent labels report raw counts as n/N. Together, the four RQ3 figures display
        {plotted_profiles_n} positive profiles covering {plotted_studies_n} studies. An additional
        {omitted_profiles_n} zero-positive profiles covering {omitted_studies_n} studies remain in the analytical
        universe but were omitted from the visual because they do not contribute positive methodological-practice
        evidence for RQ3.
        """,
    )
    write_caption(
        OUTPUT_DIR / "rq3_practice_lollipop_b_q1_v2_caption.txt",
        f"""
        RQ3B (diagnosis). Companion dot plot for diagnosis profiles. The same aggregation rule was used as in the
        prescreening and screening figure: fully specified profiles are shown individually, while partially specified
        profiles were collapsed into compact classes to preserve the counts without overextending the figure height.
        Marker position encodes within-profile frequency, and adjacent labels report raw counts as n/N.
        """,
    )
    write_caption(
        OUTPUT_DIR / "rq3_practice_lollipop_c_q1_v2_caption.txt",
        f"""
        RQ3C (monitoring/intervention). Companion dot plot for monitoring/intervention profiles using the same
        aggregation rule as the other RQ3 figures.
        """,
    )
    write_caption(
        OUTPUT_DIR / "rq3_practice_lollipop_d_q1_v2_caption.txt",
        f"""
        RQ3D (prognosis and unspecified stage). Companion dot plot for prognosis and unspecified clinical stage
        profiles. The same aggregation rule was used as in the other RQ3 figures:
        fully specified profiles are shown individually, while partially specified profiles were collapsed into compact
        classes to preserve the counts without overextending the figure height. Marker position encodes within-profile
        frequency, and adjacent labels report raw counts as n/N.
        """,
    )
    refresh_master_denominator_table()


def _self_check() -> None:
    assert normalize_signal(1) == 1
    assert normalize_signal(0) == 0


if __name__ == "__main__":
    _self_check()
    main()
