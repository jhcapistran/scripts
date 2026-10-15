from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
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

MIN_PROFILE_N = 8

STATUS_COLORS = {
    "Reported": "#2a6f77",
    "Not reported": "#c9ced3",
    "Not ascertainable": "#c79a3b",
}

PRACTICE_DISPLAY = {
    "external_validation": "External\nvalidation",
    "multisource_integration": "Multisource\nintegration",
    "xai_broad": "Explicit\nexplainability",
    "cross_site_robustness": "Cross-site\nevaluation",
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


def build_stage_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for stage in STAGE_ORDER:
        stage_df = df[df["stage_primary"].astype(str).eq(stage)]
        row = {"stage": stage, "stage_n": int(len(stage_df))}
        for field in PRACTICE_COLS:
            values = stage_df[field].astype(str).str.strip()
            row[f"{field}_reported_n"] = int(values.eq("Reported").sum())
            row[f"{field}_not_reported_n"] = int(values.eq("Not reported in assessed sources").sum())
            row[f"{field}_not_ascertainable_n"] = int(values.eq("Not ascertainable").sum())
            row[f"{field}_reported_rate"] = row[f"{field}_reported_n"] / len(stage_df)
        rows.append(row)
    return pd.DataFrame(rows)


def build_selected_profile_summary(combo: pd.DataFrame) -> pd.DataFrame:
    profiles = combo.copy()
    fully_specified = (
        profiles["stage_norm"].astype(str).ne("Not specified")
        & profiles["modality_norm"].astype(str).ne("Not specified")
        & profiles["method_norm"].astype(str).ne("Not specified")
    )
    selected = profiles[fully_specified & profiles["combo_n"].ge(MIN_PROFILE_N)].copy()
    selected["profile_group"] = f"n >= {MIN_PROFILE_N}"
    missing_stages = set(STAGE_ORDER[:-1]) - set(selected["stage_norm"].astype(str))
    representatives = (
        profiles[fully_specified & profiles["stage_norm"].astype(str).isin(missing_stages)]
        .sort_values(["stage_norm", "combo_n", "modality_norm", "method_norm"], ascending=[True, False, True, True])
        .groupby("stage_norm", observed=True, sort=False)
        .head(1)
        .copy()
    )
    representatives["profile_group"] = "Stage representative"
    selected = pd.concat([selected, representatives], ignore_index=True)
    selected["plot_stage"] = selected["stage_norm"].astype(str)
    selected["plot_modality"] = selected["modality_norm"].astype(str)
    selected["plot_method"] = selected["method_norm"].astype(str)
    selected["specification_group"] = "Fully specified profile"
    selected["stage_sort"] = pd.Categorical(selected["plot_stage"], categories=STAGE_ORDER[:-1], ordered=True)
    selected = selected.sort_values(
        ["stage_sort", "combo_n", "plot_modality", "plot_method"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)
    selected["plot_label"] = selected.apply(
        lambda row: f"{row['plot_stage']} | {row['plot_modality']} | {row['plot_method']} (n={int(row['combo_n'])})",
        axis=1,
    )
    return selected


def draw_overall_status(global_summary: pd.DataFrame, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.45), facecolor="white")
    status_specs = [("Reported", "reported_n"), ("Not reported", "not_reported_n"), ("Not ascertainable", "not_ascertainable_n")]
    for row_index, row in global_summary.iterrows():
        left = 0.0
        for status, column in status_specs:
            count = int(row[column])
            share = count / int(row["total_n"])
            ax.barh(row_index, share, left=left, height=0.62, color=STATUS_COLORS[status], edgecolor="white", linewidth=0.6)
            if share >= 0.075:
                ax.text(left + share / 2, row_index, f"{count}\n({share:.1%})", ha="center", va="center", fontsize=8, color="white" if status == "Reported" else "#20252a")
            elif count:
                above = status == "Reported"
                ax.annotate(
                    f"{count}/428 ({share:.1%})",
                    xy=(left + share / 2, row_index),
                    xytext=(0.10 if above else 0.90, row_index - 0.38 if above else row_index + 0.38),
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    arrowprops={"arrowstyle": "-", "color": "#6b737a", "linewidth": 0.7},
                    clip_on=False,
                )
            left += share
    ax.set_yticks(range(len(global_summary)), global_summary["practice"], fontsize=9)
    ax.set_ylim(len(global_summary) - 0.35, -0.65)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel("Share of studies (N=428)", fontsize=9)
    ax.grid(axis="x", color="#e3e6e8", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)
    legend_labels = {"Reported": "Reported", "Not reported": "Not reported in assessed sources", "Not ascertainable": "Not ascertainable"}
    ax.legend(handles=[Patch(facecolor=color, label=legend_labels[status]) for status, color in STATUS_COLORS.items()], frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.20), fontsize=7.5)
    fig.subplots_adjust(left=0.34, right=0.98, top=0.94, bottom=0.28)
    save_figure_variants(fig, outpath)


def draw_reported_heatmap(summary: pd.DataFrame, outpath: Path, profile_matrix: bool = False) -> None:
    rates = summary[[f"{field}_reported_rate" for field in PRACTICE_COLS]].to_numpy()
    counts = summary[[f"{field}_reported_count" if profile_matrix else f"{field}_reported_n" for field in PRACTICE_COLS]].to_numpy()
    denominators = summary["combo_n" if profile_matrix else "stage_n"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.2, 6.9 if profile_matrix else 3.8), facecolor="white")
    image = ax.imshow(rates, cmap="Blues", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    for row in range(len(summary)):
        for col in range(len(PRACTICE_COLS)):
            ax.text(col, row, f"{int(counts[row, col])}/{int(denominators[row])}", ha="center", va="center", fontsize=8.5 if profile_matrix else 9.5, color="white" if rates[row, col] >= 0.55 else "#17232b")
    ax.set_xticks(range(len(PRACTICE_COLS)), [PRACTICE_DISPLAY[field] for field in PRACTICE_COLS], fontsize=8.5)
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", length=0, pad=6)
    ax.set_yticks(range(len(summary)))
    if profile_matrix:
        short_data = {
            "Structured clinical/questionnaire data": "Clinical/questionnaire",
            "Behavioral images/video/gaze/movement": "Behavioral/gaze/movement",
            "Multiple source categories": "Multiple sources",
            "Other biomedical/digital data": "Other biomedical/digital",
        }
        short_ai = {"Machine Learning": "ML", "Deep Learning": "DL", "Hybrid": "Hybrid"}
        labels = []
        previous_stage = None
        for _, row in summary.iterrows():
            stage = str(row["plot_stage"])
            prefix = f"{stage}\n" if stage != previous_stage else ""
            labels.append(f"{prefix}{short_data.get(str(row['plot_modality']), row['plot_modality'])} · {short_ai.get(str(row['plot_method']), row['plot_method'])} (n={int(row['combo_n'])})")
            previous_stage = stage
        ax.set_yticklabels(labels, fontsize=8)
        for boundary in summary.groupby("plot_stage", observed=True, sort=False).size().cumsum().iloc[:-1]:
            ax.axhline(boundary - 0.5, color="white", linewidth=2.2)
    else:
        ax.set_yticklabels([f"{row.stage} (N={int(row.stage_n)})" for row in summary.itertuples()], fontsize=9)
        ax.axhline(4.5, color="white", linewidth=4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colorbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.045, pad=0.08, aspect=35)
    colorbar.ax.xaxis.set_major_formatter(PercentFormatter(1))
    colorbar.set_label("Studies reporting the practice", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    fig.subplots_adjust(left=0.48 if profile_matrix else 0.27, right=0.98, top=0.91, bottom=0.10 if profile_matrix else 0.18)
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

    plot_combos = build_plot_summary(combo).sort_values(
        ["stage_sort", "specification_group", "max_practice_rate", "combo_n", "positive_practice_total"],
        ascending=[True, False, False, False, False],
    ).reset_index(drop=True)
    stage_summary = build_stage_summary(df)
    selected_profiles = build_selected_profile_summary(combo)
    if len(selected_profiles) != 17 or int(selected_profiles["combo_n"].sum()) != 277:
        raise ValueError("Figure 8 selection must contain 17 profiles covering 277 studies.")

    combo.to_csv(SUPPLEMENT_DIR / "rq3_combo_summary_q1_v2.csv", index=False)
    plot_combos.to_csv(OUTPUT_DIR / "rq3_lollipop_combos_q1_v2.csv", index=False)
    stage_summary.to_csv(OUTPUT_DIR / "rq3_stage_practice_summary_q1_v2.csv", index=False)
    selected_profiles.to_csv(OUTPUT_DIR / "rq3_ai_data_stage_matrix_q1_v2.csv", index=False)
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

    draw_overall_status(global_summary, OUTPUT_DIR / "rq3_figure6_practice_reporting_q1_v2")
    draw_reported_heatmap(stage_summary, OUTPUT_DIR / "rq3_figure7_stage_practice_heatmap_q1_v2")
    draw_reported_heatmap(selected_profiles, OUTPUT_DIR / "rq3_figure8_ai_data_stage_matrix_q1_v2", profile_matrix=True)

    write_caption(
        OUTPUT_DIR / "rq3_figure6_practice_reporting_q1_v2_caption.txt",
        """
        Figure 6. Reporting status of four methodological practices across 428 included studies. Bars show the mutually exclusive adjudicated states
        Reported, Not reported in assessed sources, and Not ascertainable. Absence of reporting does not establish absence of the practice.
        """,
    )
    write_caption(
        OUTPUT_DIR / "rq3_figure7_stage_practice_heatmap_q1_v2_caption.txt",
        """
        Figure 7. Methodological practices reported across six clinical-stage categories. Cell labels give reported studies over the stage denominator;
        color encodes the corresponding percentage. Not reported and not ascertainable records remain in every denominator.
        """,
    )
    write_caption(
        OUTPUT_DIR / "rq3_figure8_ai_data_stage_matrix_q1_v2_caption.txt",
        f"""
        Figure 8. Reporting of four methodological practices across selected, fully specified clinical stage-data source-AI technique profiles. Cell labels
        give reported studies over the profile denominator; color encodes the percentage. The matrix shows profiles with n>={MIN_PROFILE_N} plus the most
        frequent fully specified profile for each otherwise unrepresented clinical stage: {len(selected_profiles)} profiles covering
        {int(selected_profiles['combo_n'].sum())} studies. It does not represent all 428 studies; the remaining studies are retained in Figures 6-7 and the
        complete analytical data. ML, machine learning; DL, deep learning; Clinical/questionnaire, structured clinical or questionnaire data;
        Behavioral/gaze/movement, behavioral images, video, gaze, or movement data. Absence of reporting does not establish absence of practice.
        """,
    )

    for letter in "abcd":
        for suffix in (".png", ".pdf", ".svg", "_caption.txt"):
            (OUTPUT_DIR / f"rq3_practice_lollipop_{letter}_q1_v2{suffix}").unlink(missing_ok=True)

    refresh_master_denominator_table()


def _self_check() -> None:
    assert len(PRACTICE_COLS) == 4


if __name__ == "__main__":
    _self_check()
    main()
