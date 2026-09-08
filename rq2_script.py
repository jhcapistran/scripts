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

STAGE_ORDER = [
    "Prescreening",
    "Screening",
    "Diagnosis",
    "Prognosis",
    "Monitoring/intervention",
    "Not specified",
]

STATUS_ORDER = [
    "Evaluated at point of use",
    "Evaluated AI-supported intervention",
    "Evaluated care-delivery platform",
    "Proposed only",
    "Research only",
]


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


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    for stale in [
        "rq2_timing_table_q1_v2.csv",
        "rq2_stage_x_preliminary_signal_q1_v2.csv",
        "rq2_preliminary_signals_for_manual_review_q1_v2.csv",
        "rq2_review_rows_q1_v2.csv",
    ]:
        stale_path = OUTPUT_DIR / stale
        if stale_path.exists():
            stale_path.unlink()

    df = load_base_df()
    df["stage_norm"] = df["stage_primary"].astype(str)
    df["rq2_status_norm"] = df["rq2_status"].astype(str)

    stage_signal = (
        pd.crosstab(df["stage_norm"], df["rq2_status_norm"])
        .reindex(
            index=STAGE_ORDER,
            columns=STATUS_ORDER,
            fill_value=0,
        )
    )

    stage_signal.to_csv(OUTPUT_DIR / "rq2_stage_x_integration_status_q1_v2.csv")

    export_cols = [
        "study_id",
        "cohort",
        "title",
        "doi",
        "year",
        "stage_primary",
        "rq2_status",
        "rq2_role",
        "decision_timing_coded",
        "observed_decision_timing",
        "source_level",
        "rationale",
    ]
    available_cols = [c for c in export_cols if c in df.columns]
    df[available_cols].to_csv(OUTPUT_DIR / "rq2_tripartite_dataset_q1_v2.csv", index=False)

    counts = df["rq2_status_norm"].value_counts().to_dict()
    evaluated_n = int(df["rq2_status_norm"].str.startswith("Evaluated").sum())

    denominator_rows = [
        {"rq": "RQ2", "subset": "all_rows", "group": "Evaluated studies", "denominator_n": int(len(df)), "notes": "Analytical universe: included reports evaluated for RQ2 from Maestro_IA_TEA_cierre_2026-09-08.xlsx."},
        {"rq": "RQ2", "subset": "integration_evaluation", "group": "Evaluated AI use", "denominator_n": evaluated_n, "notes": "Reports documenting evaluation of AI at point of use, in care-delivery, or in an AI-supported intervention."},
        {"rq": "RQ2", "subset": "integration_evaluation", "group": "Proposed only", "denominator_n": int(counts.get("Proposed only", 0)), "notes": "Reports with translational or clinical use proposed but without user/workflow evaluation."},
        {"rq": "RQ2", "subset": "integration_evaluation", "group": "Research only", "denominator_n": int(counts.get("Research only", 0)), "notes": "Reports with research-only modeling or biomarker discovery."},
    ]
    for status in STATUS_ORDER:
        denominator_rows.append({
            "rq": "RQ2",
            "subset": "adjudicated_status",
            "group": status,
            "denominator_n": int(counts.get(status, 0)),
            "notes": f"Adjudicated rq2_status in final master closure.",
        })
    pd.DataFrame(denominator_rows).to_csv(OUTPUT_DIR / "rq2_denominators_q1_v2.csv", index=False)

    fig, heat_ax = plt.subplots(figsize=(10.8, 6.8), facecolor="white")
    heat_values = stage_signal.to_numpy(dtype=float)
    img = heat_ax.imshow(heat_values, cmap="Blues", aspect="auto")

    vmin, vmax = heat_values.min(), heat_values.max()
    threshold = vmin + (vmax - vmin) * 0.52
    row_totals = heat_values.sum(axis=1)

    for row_idx in range(heat_values.shape[0]):
        for col_idx in range(heat_values.shape[1]):
            value = int(heat_values[row_idx, col_idx])
            share = 0.0 if row_totals[row_idx] == 0 else value / row_totals[row_idx]
            if value == 0:
                label = "0"
                font_color = "#888888"
            else:
                label = f"{value}\n{share:.1%}" if share < 0.999 else f"{value}\n100%"
                font_color = "white" if heat_values[row_idx, col_idx] >= threshold else "#1a1a2e"
            heat_ax.text(
                col_idx, row_idx, label,
                ha="center", va="center",
                fontsize=11, fontweight="bold",
                color=font_color,
                linespacing=1.35,
            )

    heat_ax.set_xticks(range(len(stage_signal.columns)))
    heat_ax.set_xticklabels(
        [col.replace("Evaluated ", "Evaluated\n").replace("Proposed only", "Proposed\nonly").replace("Research only", "Research\nonly") for col in stage_signal.columns],
        fontsize=11.5,
    )
    heat_ax.set_yticks(range(len(stage_signal.index)))
    heat_ax.set_yticklabels(stage_signal.index, fontsize=12)
    heat_ax.set_xlabel("Adjudicated clinical integration status", fontsize=14.5, fontweight="bold", labelpad=14)
    heat_ax.set_ylabel("Clinical stage", fontsize=14, fontweight="bold", labelpad=12)

    cbar = fig.colorbar(img, ax=heat_ax, shrink=0.75, pad=0.03)
    cbar.set_label("Report count", fontsize=12)
    cbar.ax.tick_params(labelsize=11)
    fig.tight_layout()
    save_figure_variants(fig, OUTPUT_DIR / "rq2_heatmap_and_timing_q1_v2")

    write_caption(
        OUTPUT_DIR / "rq2_heatmap_and_timing_q1_v2_caption.txt",
        f"""
        RQ2. Adjudicated clinical-integration heatmap. Distribution of clinical integration maturity across functional
        clinical stages from the final frozen master workbook (Maestro_IA_TEA_cierre_2026-09-08.xlsx). The analytical universe
        contains {len(df)} reports: {evaluated_n} reports ({evaluated_n/len(df):.1%}) with evaluated AI use ({counts.get('Evaluated at point of use', 0)} point-of-use evaluations,
        {counts.get('Evaluated AI-supported intervention', 0)} AI-supported interventions, and {counts.get('Evaluated care-delivery platform', 0)} care-delivery platform),
        {counts.get('Proposed only', 0)} reports ({counts.get('Proposed only', 0)/len(df):.1%}) proposing translational tools or workflows without direct implementation testing,
        and {counts.get('Research only', 0)} reports ({counts.get('Research only', 0)/len(df):.1%}) restricted to research modeling.
        Cell annotations report the absolute report count and the within-stage percentage.
        """,
    )


def _self_check() -> None:
    assert len(STAGE_ORDER) == 6
    assert len(STATUS_ORDER) == 5


if __name__ == "__main__":
    _self_check()
    main()
