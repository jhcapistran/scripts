from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import scienceplots


matplotlib.use("Agg")
plt.style.use(["science", "no-latex"])

BASE_DIR = Path(__file__).resolve().parent
MASTER_FILE = BASE_DIR / "Maestro_IA_TEA_cierre_2026-09-08.xlsx"
SHEET_NAME = "BASE_CIERRE"
OUTPUT_DIR = BASE_DIR / "rq2_results_q1_v2"

MATURITY_FIGURE_STEM = "rq2_integration_maturity_by_stage_q1_v2"
TIMING_FIGURE_STEM = "rq2_decision_timing_comparison_q1_v2"
LEGACY_OUTPUT_STEMS = [
    "rq2_integration_levels_and_timing_q1_v2",
    "rq2_heatmap_and_timing_q1_v2",
]

STAGE_ORDER = [
    "Prescreening",
    "Screening",
    "Diagnosis",
    "Prognosis",
    "Monitoring/intervention",
    "Not specified",
]

MATURITY_ORDER = ["Research only", "Proposed only", "Evaluated AI use"]
STATUS_ORDER = [
    "Evaluated at point of use",
    "Evaluated AI-supported intervention",
    "Evaluated care-delivery platform",
    "Proposed only",
    "Research only",
]
TIMING_ORDER = ["Pre-decision", "In-decision", "Post-decision"]

MATURITY_COLORS = {
    "Research only": "#B8C0C8",
    "Proposed only": "#0072B2",
    "Evaluated AI use": "#D55E00",
}
TIMING_COLORS = {
    "Pre-decision": "#0072B2",
    "In-decision": "#E69F00",
    "Post-decision": "#009E73",
}

ADJUDICATED_COLS = [
    "rq2_status",
    "rq2_role",
    "decision_timing_coded",
    "observed_decision_timing",
    "source_level",
    "rationale",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_figure_variants(fig: plt.Figure, stem: Path) -> None:
    for suffix in (".png", ".pdf", ".svg"):
        kwargs = {"bbox_inches": "tight"}
        if suffix == ".png":
            kwargs["dpi"] = 600
        fig.savefig(stem.with_suffix(suffix), **kwargs)
    plt.close(fig)


def write_caption(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def remove_stale_outputs() -> None:
    stale_names = [
        "rq2_timing_table_q1_v2.csv",
        "rq2_stage_x_preliminary_signal_q1_v2.csv",
        "rq2_preliminary_signals_for_manual_review_q1_v2.csv",
        "rq2_review_rows_q1_v2.csv",
    ]
    for stem in LEGACY_OUTPUT_STEMS:
        stale_names.extend(f"{stem}{suffix}" for suffix in (".png", ".pdf", ".svg", "_caption.txt"))
    for name in stale_names:
        path = OUTPUT_DIR / name
        if path.exists():
            path.unlink()


def load_base_df() -> pd.DataFrame:
    if not MASTER_FILE.exists():
        raise FileNotFoundError(f"Master file not found: {MASTER_FILE}")
    df = pd.read_excel(MASTER_FILE, sheet_name=SHEET_NAME, skiprows=3)
    df = df[df["include_main"] == 1].copy()
    if len(df) != 428:
        raise ValueError(f"Expected 428 included studies in BASE_CIERRE, found {len(df)}")
    return df


def maturity_from_status(status: object) -> str:
    status_text = str(status)
    if status_text.startswith("Evaluated"):
        return "Evaluated AI use"
    if status_text in ("Research only", "Proposed only"):
        return status_text
    raise ValueError(f"Unexpected rq2_status: {status_text}")


def require_expected_counts(df: pd.DataFrame) -> None:
    expected = {"Research only": 311, "Proposed only": 98, "Evaluated AI use": 19}
    actual = df["integration_maturity"].value_counts().reindex(MATURITY_ORDER, fill_value=0).to_dict()
    if actual != expected:
        detail = (
            df.loc[:, ["study_id", "title", "stage_primary", "rq2_status", "integration_maturity"]]
            .sort_values(["integration_maturity", "study_id"])
            .to_string(index=False)
        )
        raise ValueError(f"RQ2 maturity counts changed. Expected {expected}, found {actual}.\n{detail}")


def count_series(series: pd.Series, order: list[str], count_name: str = "n") -> pd.DataFrame:
    return (
        series.value_counts()
        .reindex(order, fill_value=0)
        .rename_axis(series.name)
        .reset_index(name=count_name)
    )


def plot_stacked_percent(
    ax: plt.Axes,
    counts: pd.DataFrame,
    order: list[str],
    colors: dict[str, str],
    inside_threshold: float,
) -> None:
    totals = counts.sum(axis=1)
    percentages = counts.div(totals, axis=0).mul(100)
    left = pd.Series(0.0, index=counts.index)
    for category in order:
        bars = ax.barh(
            counts.index,
            percentages[category],
            left=left,
            height=0.62,
            color=colors[category],
            edgecolor="white",
            linewidth=0.8,
            label=category,
        )
        for idx, bar in enumerate(bars):
            value = int(counts.iloc[idx][category])
            share = float(percentages.iloc[idx][category])
            if not value:
                continue
            label = f"{value} ({share:.1f}%)"
            center = float(left.iloc[idx]) + share / 2
            if share >= inside_threshold:
                text_color = "#1F2933" if category == "Research only" else "white"
                ax.text(center, bar.get_y() + bar.get_height() / 2, label, ha="center", va="center", color=text_color, fontsize=8.5, fontweight="semibold")
            else:
                endpoint = float(left.iloc[idx]) + share
                terminal_segment = endpoint >= 99
                if terminal_segment:
                    ax.annotate(
                        label,
                        xy=(endpoint, bar.get_y() + bar.get_height() / 2),
                        xytext=(102.5, bar.get_y() + bar.get_height() / 2),
                        ha="left",
                        va="center",
                        fontsize=8.2,
                        color="#374151",
                        arrowprops={"arrowstyle": "-", "color": "#6B7280", "lw": 0.7},
                        clip_on=False,
                    )
                else:
                    ax.annotate(
                        label,
                        xy=(center, bar.get_y() + bar.get_height()),
                        xytext=(center, bar.get_y() + bar.get_height() + 0.24),
                        ha="center",
                        va="center",
                        fontsize=8.2,
                        color="#374151",
                        arrowprops={"arrowstyle": "-", "color": "#6B7280", "lw": 0.7},
                        clip_on=False,
                    )
        left += percentages[category]


def style_percent_axis(ax: plt.Axes) -> None:
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100], labels=["0%", "25%", "50%", "75%", "100%"])
    ax.grid(axis="x", color="#D9DEE3", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="x", which="both", top=False)
    ax.tick_params(axis="y", which="both", left=False, right=False)


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    remove_stale_outputs()

    df = load_base_df()
    df["stage_norm"] = df["stage_primary"].astype(str)
    df["rq2_status_norm"] = df["rq2_status"].astype(str)
    df["integration_maturity"] = df["rq2_status"].map(maturity_from_status)
    require_expected_counts(df)

    proposed_or_evaluated = df[df["integration_maturity"].isin(["Proposed only", "Evaluated AI use"])].copy()
    proposed_only = df[df["integration_maturity"] == "Proposed only"].copy()
    evaluated = df[df["integration_maturity"] == "Evaluated AI use"].copy()

    stage_maturity = (
        pd.crosstab(df["stage_norm"], df["integration_maturity"])
        .reindex(index=STAGE_ORDER, columns=MATURITY_ORDER, fill_value=0)
    )
    stage_status = (
        pd.crosstab(df["stage_norm"], df["rq2_status_norm"])
        .reindex(index=STAGE_ORDER, columns=STATUS_ORDER, fill_value=0)
    )
    role_counts = count_series(proposed_or_evaluated["rq2_role"], sorted(proposed_or_evaluated["rq2_role"].dropna().unique()), "n")
    timing_coded = count_series(proposed_or_evaluated["decision_timing_coded"], TIMING_ORDER, "n")
    timing_proposed = count_series(proposed_only["decision_timing_coded"], TIMING_ORDER, "n")
    timing_observed = count_series(evaluated["observed_decision_timing"], TIMING_ORDER, "n")

    stage_maturity.to_csv(OUTPUT_DIR / "rq2_stage_x_integration_maturity_q1_v2.csv")
    stage_status.to_csv(OUTPUT_DIR / "rq2_stage_x_integration_status_q1_v2.csv")
    role_counts.to_csv(OUTPUT_DIR / "rq2_role_proposed_evaluated_q1_v2.csv", index=False)
    timing_coded.to_csv(OUTPUT_DIR / "rq2_decision_timing_coded_proposed_evaluated_q1_v2.csv", index=False)
    timing_observed.to_csv(OUTPUT_DIR / "rq2_observed_decision_timing_evaluated_q1_v2.csv", index=False)
    df[ADJUDICATED_COLS].to_csv(OUTPUT_DIR / "rq2_adjudicated_columns_q1_v2.csv", index=False)

    export_cols = [
        "study_id",
        "cohort",
        "title",
        "doi",
        "year",
        "stage_primary",
        *ADJUDICATED_COLS,
    ]
    df[[c for c in export_cols if c in df.columns]].to_csv(OUTPUT_DIR / "rq2_tripartite_dataset_q1_v2.csv", index=False)

    maturity_counts = df["integration_maturity"].value_counts().reindex(MATURITY_ORDER, fill_value=0)
    status_counts = df["rq2_status_norm"].value_counts().to_dict()
    evaluated_n = int(maturity_counts["Evaluated AI use"])

    denominator_rows = [
        {"rq": "RQ2", "subset": "all_rows", "group": "All included", "denominator_n": int(len(df)), "notes": "Included reports from BASE_CIERRE where include_main == 1."},
        {"rq": "RQ2", "subset": "integration_maturity", "group": "Research only", "denominator_n": int(maturity_counts["Research only"]), "notes": "Adjudicated research-only modeling or discovery; not observed clinical AI use."},
        {"rq": "RQ2", "subset": "integration_maturity", "group": "Proposed only", "denominator_n": int(maturity_counts["Proposed only"]), "notes": "Adjudicated proposed clinical or translational use without observed workflow evaluation."},
        {"rq": "RQ2", "subset": "integration_maturity", "group": "Evaluated AI use", "denominator_n": evaluated_n, "notes": "Adjudicated evaluated point-of-use, intervention, or care-delivery AI use."},
        {"rq": "RQ2", "subset": "role_and_coded_timing", "group": "Proposed/Evaluated", "denominator_n": int(len(proposed_or_evaluated)), "notes": "Reports with proposed or evaluated clinical/translational use; timing is coded from adjudicated decision_timing_coded."},
        {"rq": "RQ2", "subset": "observed_timing", "group": "Evaluated AI use", "denominator_n": evaluated_n, "notes": "Only reports with evaluated AI use; timing is observed_decision_timing."},
    ]
    for status in STATUS_ORDER:
        denominator_rows.append({
            "rq": "RQ2",
            "subset": "adjudicated_status",
            "group": status,
            "denominator_n": int(status_counts.get(status, 0)),
            "notes": "Adjudicated rq2_status in final master closure.",
        })
    pd.DataFrame(denominator_rows).to_csv(OUTPUT_DIR / "rq2_denominators_q1_v2.csv", index=False)

    stage_labels = [f"{stage}  (n={int(stage_maturity.loc[stage].sum())})" for stage in STAGE_ORDER]
    maturity_plot = stage_maturity.copy()
    maturity_plot.index = stage_labels
    fig, ax = plt.subplots(figsize=(7.4, 4.4), facecolor="white")
    plot_stacked_percent(ax, maturity_plot, MATURITY_ORDER, MATURITY_COLORS, inside_threshold=8)
    style_percent_axis(ax)
    ax.set_xlabel("Within-stage proportion (%)")
    ax.invert_yaxis()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper left", bbox_to_anchor=(0.28, 0.98), frameon=False, fontsize=9, handlelength=1.4, columnspacing=1.5)
    fig.subplots_adjust(left=0.28, right=0.86, top=0.86, bottom=0.15)
    save_figure_variants(fig, OUTPUT_DIR / MATURITY_FIGURE_STEM)

    write_caption(
        OUTPUT_DIR / f"{MATURITY_FIGURE_STEM}_caption.txt",
        f"Figure 4. Integration maturity across clinical stages. Bars show the within-stage distribution of Research only, "
        f"Proposed only, and Evaluated AI use among all {len(df)} included studies. Stage totals are shown beside the labels; "
        "segment labels give counts and within-stage percentages. Proposed only denotes a stated clinical or translational "
        "application without observed workflow evaluation, whereas Evaluated AI use denotes point-of-use, intervention, or "
        "care-delivery evaluation in the assessed sources.",
    )

    timing_matrix = pd.DataFrame(
        [timing_proposed.set_index("decision_timing_coded")["n"], timing_observed.set_index("observed_decision_timing")["n"]],
        index=[f"Proposed only (n={len(proposed_only)})", f"Evaluated AI use (n={len(evaluated)})"],
    )[TIMING_ORDER]
    fig, ax = plt.subplots(figsize=(7.5, 2.9), facecolor="white")
    plot_stacked_percent(ax, timing_matrix, TIMING_ORDER, TIMING_COLORS, inside_threshold=9)
    style_percent_axis(ax)
    ax.set_xlabel("Within-group proportion (%)")
    ax.set_ylim(-0.5, 1.72)
    ax.invert_yaxis()
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper left", bbox_to_anchor=(0.27, 0.85), frameon=False, fontsize=9, handlelength=1.4, columnspacing=1.5)
    fig.subplots_adjust(left=0.27, right=0.95, top=0.65, bottom=0.23)
    save_figure_variants(fig, OUTPUT_DIR / TIMING_FIGURE_STEM)

    write_caption(
        OUTPUT_DIR / f"{TIMING_FIGURE_STEM}_caption.txt",
        "Figure 5. Decision timing by integration maturity. Mutually exclusive groups are compared: Proposed only studies "
        f"(n={len(proposed_only)}) use decision timing coded according to the intended function, whereas Evaluated AI use "
        f"studies (n={evaluated_n}) use the timing of evaluated use observed in the assessed sources. Segment labels give "
        "counts and within-group percentages. The comparison does not represent temporal progression, system transition, "
        "or a causal relationship.",
    )


def _self_check() -> None:
    assert len(STAGE_ORDER) == 6
    assert len(MATURITY_ORDER) == 3
    assert len(TIMING_ORDER) == 3


if __name__ == "__main__":
    _self_check()
    main()
