from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import scienceplots


matplotlib.use("Agg")
plt.style.use(["science", "no-latex"])

BASE_DIR = Path(__file__).resolve().parent
MASTER_FILE = BASE_DIR / "Maestro_IA_TEA_cierre_2026-09-08.xlsx"
SHEET_NAME = "BASE_CIERRE"
OUTPUT_DIR = BASE_DIR / "rq3_results_q1_v2"
SUPPLEMENT_DIR = OUTPUT_DIR / "supplement"
MASTER_DENOMINATOR_FILE = BASE_DIR / "rq_denominators_q1_v2.csv"

PRACTICE_COLS = {
    "external_validation": ("External validation", "#d95f02", "X"),
    "multisource_integration": ("Multisource integration", "#1f78b4", "s"),
    "xai_broad": ("Explicit explainability (broad)", "#1b9e77", "D"),
    "cross_site_robustness": ("Cross-site evaluation", "#7570b3", "^"),
}

STAGE_ORDER = [
    "Prescreening",
    "Screening",
    "Diagnosis",
    "Prognosis",
    "Monitoring/intervention",
    "Not specified",
]

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

MIN_COMBO_N_FOR_FIGURE = 1
TOP_COMBOS_FOR_FIGURE = 999

STATUS_COLORS = {
    "Reported": "#1b9e77",
    "Not reported in assessed sources": "#d1d5db",
    "Not ascertainable": "#e6ab02",
}


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


def ordered_categories(observed: list[str], preferred: list[str]) -> list[str]:
    ordered = [item for item in preferred if item in observed]
    extras = sorted(item for item in observed if item not in preferred)
    return ordered + extras


def build_combo_summary(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["stage_norm"] = work["stage_primary"].astype(str)
    work["modality_norm"] = work["data_source_primary"].astype(str)
    work["method_norm"] = work["ai_type"].astype(str)

    for col in PRACTICE_COLS:
        values = work[col].astype(str).str.strip()
        work[f"{col}_reported"] = values.eq("Reported").astype(int)
        work[f"{col}_not_reported"] = values.eq("Not reported in assessed sources").astype(int)
        work[f"{col}_not_ascertainable"] = values.eq("Not ascertainable").astype(int)

    aggregations = {"combo_n": ("study_id", "size")}
    for col in PRACTICE_COLS:
        for status in ("reported", "not_reported", "not_ascertainable"):
            aggregations[f"{col}_{status}_count"] = (f"{col}_{status}", "sum")

    combo = (
        work.groupby(["stage_norm", "modality_norm", "method_norm"], dropna=False)
        .agg(**aggregations)
        .reset_index()
    )

    for col in PRACTICE_COLS:
        combo[f"{col}_count"] = combo[f"{col}_reported_count"]
        for status in ("reported", "not_reported", "not_ascertainable"):
            combo[f"{col}_{status}_rate"] = combo[f"{col}_{status}_count"] / combo["combo_n"]
        combo[f"{col}_rate"] = combo[f"{col}_reported_rate"]

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
        for status in ("reported", "not_reported", "not_ascertainable"):
            agg_map[f"{col}_{status}_count"] = "sum"
    plot = work.groupby(group_cols, dropna=False).agg(agg_map).reset_index()
    for col in PRACTICE_COLS:
        plot[f"{col}_count"] = plot[f"{col}_reported_count"]
        for status in ("reported", "not_reported", "not_ascertainable"):
            plot[f"{col}_{status}_rate"] = plot[f"{col}_{status}_count"] / plot["combo_n"]
        plot[f"{col}_rate"] = plot[f"{col}_reported_rate"]
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
        reported_n = int(df[col].astype(str).str.strip().eq("Reported").sum())
        not_rep_n = int(df[col].astype(str).str.strip().eq("Not reported in assessed sources").sum())
        not_asc_n = int(df[col].astype(str).str.strip().eq("Not ascertainable").sum())
        rows.append(
            {
                "practice_field": col,
                "practice": label,
                "reported_n": reported_n,
                "not_reported_n": not_rep_n,
                "not_ascertainable_n": not_asc_n,
                "total_n": total_n,
                "reported_rate": reported_n / total_n if total_n else 0.0,
            }
        )
    return pd.DataFrame(rows)


def select_plot_combos(combo: pd.DataFrame) -> pd.DataFrame:
    plot_df = build_plot_summary(combo)
    plot_df = plot_df[plot_df["combo_n"] >= MIN_COMBO_N_FOR_FIGURE].copy()
    return plot_df.head(TOP_COMBOS_FOR_FIGURE).sort_values(
        ["stage_sort", "specification_group", "max_practice_rate", "combo_n", "positive_practice_total"],
        ascending=[True, False, False, False, False],
    ).reset_index(drop=True)


def draw_compact_dotplot(
    combo: pd.DataFrame,
    outpath: Path,
    panel_specs: list[tuple[str, list[str]]],
    layout: dict[str, float] | None = None,
) -> None:
    plot_df = select_plot_combos(combo)
    if plot_df.empty:
        raise ValueError("No Q3 combinations available to plot.")
    layout = layout or {}

    panel_data = [(title, plot_df[plot_df["plot_stage"].isin(stages)].reset_index(drop=True)) for title, stages in panel_specs]
    panel_data = [(title, df_stage) for title, df_stage in panel_data if not df_stage.empty]

    if not panel_data:
        return

    title, stage_df = panel_data[0]
    total_rows = len(stage_df)
    fig_height = max(layout.get("min_height", 5.8), layout.get("row_height", 0.36) * total_rows + 1.8)
    fig, axes = plt.subplots(1, len(PRACTICE_COLS), sharey=True, figsize=(19, fig_height), facecolor="white")
    y_positions = list(range(total_rows))

    for ax, (col, (practice_label, _, _)) in zip(axes, PRACTICE_COLS.items()):
        for row_idx, row in stage_df.iterrows():
            if int(row["positive_practice_total"]) == 0:
                ax.axhspan(row_idx - 0.48, row_idx + 0.48, color="#fff7d6", zorder=0)
            left = 0.0
            counts = []
            for status, suffix in (
                ("Reported", "reported"),
                ("Not reported in assessed sources", "not_reported"),
                ("Not ascertainable", "not_ascertainable"),
            ):
                count = int(row[f"{col}_{suffix}_count"])
                rate = float(row[f"{col}_{suffix}_rate"])
                ax.barh(row_idx, rate, left=left, height=0.72, color=STATUS_COLORS[status], edgecolor="white", linewidth=0.4)
                left += rate
                counts.append(count)
            ax.text(1.02, row_idx, "/".join(map(str, counts)), va="center", ha="left", fontsize=7.2)

        ax.set_title(practice_label, fontsize=11.5, fontweight="bold")
        ax.set_xlim(0, 1.32)
        ax.set_ylim(-0.5, total_rows - 0.5)
        ax.set_xticks([0, 0.5, 1.0], ["0%", "50%", "100%"])
        ax.set_xlabel("Share of profile", fontsize=9.5)
        ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.invert_yaxis()
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)

    labels = [f"{'○ ' if int(row['positive_practice_total']) == 0 else ''}{row['plot_label']}" for _, row in stage_df.iterrows()]
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(labels, fontsize=8.5)
    axes[0].set_ylabel("Profile (n = reports); ○ = no practice reported", fontsize=10.5, labelpad=18)
    for ax in axes[1:]:
        ax.tick_params(axis="y", labelleft=False)

    fig.suptitle(title, fontsize=14.5, fontweight="bold", y=0.985)
    legend_handles = [Patch(facecolor=color, label=status) for status, color in STATUS_COLORS.items()]
    fig.legend(
        handles=legend_handles,
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.64, 0.01),
        fontsize=10.5,
        title="Cell labels: Reported / Not reported / Not ascertainable",
        title_fontsize=9.5,
    )
    fig.subplots_adjust(left=0.34, right=0.99, top=0.90, bottom=layout.get("bottom_margin", 0.15), wspace=0.16)
    save_figure_variants(fig, outpath)


def refresh_master_denominator_table() -> None:
    frames = []
    for filename in ("rq1_denominators_q1_v2.csv", "rq2_denominators_q1_v2.csv", "rq3_denominators_q1_v2.csv"):
        path = BASE_DIR / ("rq1_results_q1_v2" if filename.startswith("rq1") else "rq2_results_q1_v2" if filename.startswith("rq2") else "rq3_results_q1_v2") / filename
        if path.exists():
            frames.append(pd.read_csv(path))

    screening_rows = pd.DataFrame([
        {
            "rq": "Screening",
            "subset": "candidate_pool",
            "group": "Historical register",
            "denominator_n": 276,
            "notes": "Historical reports retained in Maestro_IA_TEA_cierre_2026-09-08.xlsx.",
        },
        {
            "rq": "Screening",
            "subset": "candidate_pool",
            "group": "Update reports assessed",
            "denominator_n": 178,
            "notes": "New candidate reports initially included from the update.",
        },
        {
            "rq": "Screening",
            "subset": "candidate_pool",
            "group": "Combined incoming register",
            "denominator_n": 454,
            "notes": "276 historical plus 178 update reports.",
        },
        {
            "rq": "Screening",
            "subset": "exclusions",
            "group": "Closure audit exclusions",
            "denominator_n": 26,
            "notes": "23 historical and 3 update reports excluded upon audit (EXCLUSIONES_CIERRE).",
        },
        {
            "rq": "Screening",
            "subset": "final_corpus",
            "group": "Included main synthesis",
            "denominator_n": 428,
            "notes": "Frozen analytical synthesis corpus: 253 historical and 175 update reports (BASE_CIERRE include_main=1).",
        },
    ])

    all_frames = [screening_rows] + frames
    pd.concat(all_frames, ignore_index=True).to_csv(MASTER_DENOMINATOR_FILE, index=False)


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    ensure_dir(SUPPLEMENT_DIR)

    df = load_base_df()
    combo = build_combo_summary(df)
    global_summary = build_global_summary(df)

    plot_combos = select_plot_combos(combo)

    combo.to_csv(SUPPLEMENT_DIR / "rq3_combo_summary_q1_v2.csv", index=False)
    plot_combos.to_csv(OUTPUT_DIR / "rq3_lollipop_combos_q1_v2.csv", index=False)
    (OUTPUT_DIR / "rq3_omitted_zero_profiles_q1_v2.csv").unlink(missing_ok=True)
    global_summary.to_csv(OUTPUT_DIR / "rq3_global_practice_summary_q1_v2.csv", index=False)

    denominator_rows = [
        {
            "rq": "RQ3",
            "subset": "all_rows",
            "group": "Evaluated studies",
            "denominator_n": int(len(df)),
            "notes": "Analytical universe: included reports evaluated for RQ3 from Maestro_IA_TEA_cierre_2026-09-08.xlsx.",
        },
    ]

    for _, row in global_summary.iterrows():
        denominator_rows.append(
            {
                "rq": "RQ3",
                "subset": "practice_signal",
                "group": row["practice"],
                "denominator_n": int(row["total_n"]),
                "notes": f"Reported={int(row['reported_n'])}; Not reported={int(row['not_reported_n'])}; Not ascertainable={int(row['not_ascertainable_n'])}; Rate={row['reported_rate']:.1%}.",
            }
        )

    for field, label in [
        ("xai_strict", "Formal attribution/explanation (strict)"),
        ("multisite_dataset", "Multisite data availability"),
        ("prospective_evaluation", "Prospective AI evaluation"),
    ]:
        if field in df.columns:
            rep_n = int(df[field].astype(str).str.strip().eq("Reported").sum())
            denominator_rows.append({
                "rq": "RQ3",
                "subset": "additional_practice_signal",
                "group": label,
                "denominator_n": int(len(df)),
                "notes": f"Reported={rep_n}; Rate={rep_n/len(df):.1%}; field {field} from BASE_CIERRE.",
            })

    pd.DataFrame(denominator_rows).to_csv(OUTPUT_DIR / "rq3_denominators_q1_v2.csv", index=False)

    plotted_profiles_n = int(len(plot_combos))
    plotted_studies_n = int(plot_combos["combo_n"].sum()) if not plot_combos.empty else 0
    zero_profiles = plot_combos[plot_combos["positive_practice_total"] == 0]
    zero_profiles_n = int(len(zero_profiles))
    zero_studies_n = int(zero_profiles["combo_n"].sum())

    figure_specs = [
        ("rq3_practice_lollipop_a_q1_v2", "A. Prescreening and screening", ["Prescreening", "Screening"], {"row_height": 0.46, "base_height": 1.0, "min_height": 5.8}),
        ("rq3_practice_lollipop_b_q1_v2", "B. Diagnosis", ["Diagnosis"], {"row_height": 0.47, "base_height": 1.02, "min_height": 6.0}),
        ("rq3_practice_lollipop_c_q1_v2", "C. Monitoring/intervention", ["Monitoring/intervention"], {"row_height": 0.50, "base_height": 1.1, "min_height": 6.0, "bottom_margin": 0.24, "legend_y": 0.02}),
        ("rq3_practice_lollipop_d_q1_v2", "D. Prognosis and unspecified stage", ["Prognosis", "Clinical stage not specified"], {"row_height": 0.40, "base_height": 0.9, "min_height": 4.6, "bottom_margin": 0.25, "xlabel_y": 0.105, "legend_y": 0.01}),
    ]
    for stem, title, stages, layout in figure_specs:
        draw_compact_dotplot(combo, OUTPUT_DIR / stem, [(title, stages)], layout)

    write_caption(
        OUTPUT_DIR / "rq3_practice_lollipop_a_q1_v2_caption.txt",
        f"""
        RQ3A (prescreening and screening). Stacked bars show the adjudicated status of four methodological practices within combinations of
        clinical stage, data source, and AI technique in ASD studies from Maestro_IA_TEA_cierre_2026-09-08.xlsx. Each row represents one fully specified profile
        or one aggregated partially specified profile. Green, gray, and amber encode Reported, Not reported in assessed sources, and Not ascertainable,
        respectively; labels give the three raw counts in that order and the row label gives the profile denominator. Profiles with unspecified data source,
        AI technique, or clinical stage are collapsed into compact classes without duplicating reports. Together, the four RQ3 figures display
        {plotted_profiles_n} profiles covering all {plotted_studies_n} reports, including {zero_profiles_n} profiles covering {zero_studies_n} reports
        in which none of the four practices was reported (marked with an open circle and a pale background).
        """,
    )
    write_caption(
        OUTPUT_DIR / "rq3_practice_lollipop_b_q1_v2_caption.txt",
        f"""
        RQ3B (diagnosis). Companion dot plot for diagnosis profiles from Maestro_IA_TEA_cierre_2026-09-08.xlsx. The same aggregation rule was used as in the
        prescreening and screening figure: fully specified profiles are shown individually, while partially specified
        profiles were collapsed into compact classes to preserve the counts without overextending the figure height.
        Bars and labels distinguish Reported, Not reported in assessed sources, and Not ascertainable for every practice.
        """,
    )
    write_caption(
        OUTPUT_DIR / "rq3_practice_lollipop_c_q1_v2_caption.txt",
        f"""
        RQ3C (monitoring/intervention). Companion dot plot for monitoring/intervention profiles from Maestro_IA_TEA_cierre_2026-09-08.xlsx using the same
        aggregation rule as the other RQ3 figures. The legend and count labels are separated from the plotting
        area to avoid overlap.
        """,
    )
    write_caption(
        OUTPUT_DIR / "rq3_practice_lollipop_d_q1_v2_caption.txt",
        f"""
        RQ3D (prognosis and unspecified stage). Companion dot plot for prognosis and unspecified clinical stage
        profiles from Maestro_IA_TEA_cierre_2026-09-08.xlsx. The same aggregation rule was used as in the other RQ3 figures:
        fully specified profiles are shown individually, while partially specified profiles were collapsed into compact
        classes to preserve the counts without overextending the figure height. Marker position encodes within-profile
        frequency, and adjacent labels report raw counts as n/N.
        """,
    )

    refresh_master_denominator_table()


def _self_check() -> None:
    assert len(PRACTICE_COLS) == 4


if __name__ == "__main__":
    _self_check()
    main()
