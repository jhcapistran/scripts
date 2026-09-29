from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots


matplotlib.use("Agg")
plt.style.use(["science", "no-latex"])

BASE_DIR = Path(__file__).resolve().parent
MASTER_FILE = BASE_DIR / "Maestro_IA_TEA_cierre_2026-09-08.xlsx"
SHEET_NAME = "BASE_CIERRE"
OUTPUT_DIR = BASE_DIR / "rq2_results_q1_v2"

FIGURE_STEM = "rq2_integration_levels_and_timing_q1_v2"
LEGACY_OUTPUT_STEMS = ["rq2_heatmap_and_timing_q1_v2"]

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


def add_bar_labels(ax: plt.Axes, bars, total: int, horizontal: bool = False) -> None:
    for bar in bars:
        value = int(bar.get_width() if horizontal else bar.get_height())
        share = value / total if total else 0
        label = f"{value} ({share:.1%})"
        if horizontal:
            ax.text(value + max(total * 0.015, 0.5), bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=9.5)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, value + max(total * 0.015, 0.5), label, ha="center", fontsize=10)


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    remove_stale_outputs()

    df = load_base_df()
    df["stage_norm"] = df["stage_primary"].astype(str)
    df["rq2_status_norm"] = df["rq2_status"].astype(str)
    df["integration_maturity"] = df["rq2_status"].map(maturity_from_status)
    require_expected_counts(df)

    proposed_or_evaluated = df[df["integration_maturity"].isin(["Proposed only", "Evaluated AI use"])].copy()
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

    fig = plt.figure(figsize=(12.5, 10.5), facecolor="white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.2, 1], height_ratios=[1.12, 1], hspace=0.42, wspace=0.36)

    heat_ax = fig.add_subplot(gs[0, 0])
    heat_values = stage_maturity.to_numpy(dtype=float)
    img = heat_ax.imshow(heat_values, cmap="Blues", aspect="auto")
    threshold = heat_values.min() + (heat_values.max() - heat_values.min()) * 0.52
    row_totals = heat_values.sum(axis=1)
    for row_idx in range(heat_values.shape[0]):
        for col_idx in range(heat_values.shape[1]):
            value = int(heat_values[row_idx, col_idx])
            share = 0.0 if row_totals[row_idx] == 0 else value / row_totals[row_idx]
            label = f"{value}\n{share:.1%}" if value else "0"
            color = "white" if value and heat_values[row_idx, col_idx] >= threshold else "#1a1a2e"
            heat_ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=9.5, fontweight="bold", color=color, linespacing=1.25)
    heat_ax.set_xticks(range(len(MATURITY_ORDER)))
    heat_ax.set_xticklabels(["Research\nonly", "Proposed\nonly", "Evaluated\nAI use"], fontsize=10)
    heat_ax.set_yticks(range(len(STAGE_ORDER)))
    heat_ax.set_yticklabels(STAGE_ORDER, fontsize=10)
    heat_ax.set_title("A. Integration maturity x clinical stage (n=428)", loc="left", fontsize=12.5, fontweight="bold")
    heat_ax.set_xlabel("Adjudicated integration maturity", fontsize=11)
    heat_ax.set_ylabel("Clinical stage", fontsize=11)
    cbar = fig.colorbar(img, ax=heat_ax, shrink=0.72, pad=0.03)
    cbar.set_label("Report count", fontsize=10)

    role_ax = fig.add_subplot(gs[0, 1])
    role_plot = role_counts.sort_values("n", ascending=True)
    bars = role_ax.barh(role_plot["rq2_role"], role_plot["n"], color="#4c78a8")
    add_bar_labels(role_ax, bars, len(proposed_or_evaluated), horizontal=True)
    role_ax.set_title("B. Functional AI role, Proposed/Evaluated (n=117)", loc="left", fontsize=12.5, fontweight="bold")
    role_ax.set_xlabel("Reports")
    role_ax.set_xlim(0, max(role_plot["n"]) * 1.28)
    role_ax.tick_params(axis="y", labelsize=9.5)
    role_ax.spines[["top", "right"]].set_visible(False)

    coded_ax = fig.add_subplot(gs[1, 0])
    bars = coded_ax.bar(timing_coded["decision_timing_coded"], timing_coded["n"], color="#59a14f")
    add_bar_labels(coded_ax, bars, len(proposed_or_evaluated))
    coded_ax.set_title("C. Coded decision timing, Proposed/Evaluated (n=117)", loc="left", fontsize=12.5, fontweight="bold")
    coded_ax.set_ylabel("Reports")
    coded_ax.set_ylim(0, max(timing_coded["n"]) * 1.22)
    coded_ax.spines[["top", "right"]].set_visible(False)

    observed_ax = fig.add_subplot(gs[1, 1])
    bars = observed_ax.bar(timing_observed["observed_decision_timing"], timing_observed["n"], color="#e15759")
    add_bar_labels(observed_ax, bars, len(evaluated))
    observed_ax.set_title("D. Observed decision timing, Evaluated only (n=19)", loc="left", fontsize=12.5, fontweight="bold")
    observed_ax.set_ylabel("Reports")
    observed_ax.set_ylim(0, max(timing_observed["n"]) * 1.25)
    observed_ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("RQ2. Clinical integration maturity, AI role, and decision timing", fontsize=15, fontweight="bold", y=0.995)
    fig.text(
        0.5,
        0.01,
        "Panels B-C use adjudicated proposed/evaluated clinical or translational use; Panel D is restricted to observed timing in evaluated AI use.",
        ha="center",
        fontsize=10.5,
    )
    save_figure_variants(fig, OUTPUT_DIR / FIGURE_STEM)

    write_caption(
        OUTPUT_DIR / f"{FIGURE_STEM}_caption.txt",
        f"""
        Figure 4. RQ2 clinical integration levels and timing. Panel A shows integration maturity by clinical stage for all
        {len(df)} included studies from BASE_CIERRE: Research only = {maturity_counts['Research only']},
        Proposed only = {maturity_counts['Proposed only']}, and Evaluated AI use = {evaluated_n}. Panel B summarizes
        the adjudicated functional role of AI among Proposed/Evaluated studies (n={len(proposed_or_evaluated)}). Panel C
        reports adjudicated decision_timing_coded in the same Proposed/Evaluated subset. Panel D is deliberately restricted
        to observed_decision_timing among the {evaluated_n} Evaluated AI use studies, distinguishing observed timing from
        proposed or coded timing. No values are inferred from stage_primary or automated rules.
        """,
    )


def _self_check() -> None:
    assert len(STAGE_ORDER) == 6
    assert len(MATURITY_ORDER) == 3
    assert len(TIMING_ORDER) == 3


if __name__ == "__main__":
    _self_check()
    main()
