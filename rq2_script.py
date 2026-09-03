from __future__ import annotations

import datetime as dt
from pathlib import Path
import re

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots


matplotlib.use("Agg")
plt.style.use(["science", "no-latex"])

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "analysis_dataset_q1_v2.xlsx"
SHEET_NAME = "rq1_rq2_graph_ready"
OUTPUT_DIR = BASE_DIR / "rq2_results_q1_v2"

STAGE_ORDER = [
    "Prescreening",
    "Screening",
    "Diagnosis",
    "Prognosis",
    "Monitoring/intervention",
    "Not specified",
]

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
    return re.sub(r"\s+", " ", str(cleaned).casefold().replace("\n", " "))


def normalize_category(value: object, mapping: dict[str, str], fallback: str = "Not specified") -> str:
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
    if isinstance(value, dt.datetime):
        return value.date() == dt.date(1900, 1, 1)
    if isinstance(value, dt.time):
        return False
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
        return "Present"
    if value is False:
        return "Absent"
    return "Uncoded"


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


def ordered_categories(observed: list[str], preferred: list[str]) -> list[str]:
    ordered = [item for item in preferred if item in observed]
    extras = sorted(item for item in observed if item not in preferred)
    return ordered + extras


def load_base_df() -> pd.DataFrame:
    df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME).copy()
    df["row_id"] = range(1, len(df) + 1)
    return df


def normalize_common_fields(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["stage_norm"] = work["stage_primary"].apply(normalize_stage)
    return work


def derive_q2_signal_state(row: pd.Series) -> str:
    abs_signal = normalize_bool_signal(row.get("q2_candidate_abstract"))
    terms_signal = normalize_bool_signal(row.get("q2_candidate_terms"))
    if abs_signal is True or terms_signal is True:
        return "Present"
    if abs_signal is False or terms_signal is False:
        return "Absent"
    return "Uncoded"


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    for stale in ["rq2_timing_table_q1_v2.csv", "rq2_stage_x_integration_q1_v2.csv", "rq2_review_rows_q1_v2.csv"]:
        stale_path = OUTPUT_DIR / stale
        if stale_path.exists():
            stale_path.unlink()
    df = normalize_common_fields(load_base_df())
    df["q2_abstract_bool"] = df["q2_candidate_abstract"].map(normalize_bool_signal)
    df["q2_terms_bool"] = df["q2_candidate_terms"].map(normalize_bool_signal)
    df["q2_signal_state"] = df.apply(derive_q2_signal_state, axis=1)
    positive_mask = df["q2_signal_state"].eq("Present")
    df["q2_signal_pattern"] = df.apply(
        lambda row: f"abstract={status_from_bool(row['q2_abstract_bool'])}; terms={status_from_bool(row['q2_terms_bool'])}",
        axis=1,
    )
    positive_df = df.loc[positive_mask].copy()
    stage_signal = (
        pd.crosstab(df["stage_norm"], df["q2_signal_state"])
        .reindex(index=ordered_categories(df["stage_norm"].unique().tolist(), STAGE_ORDER), columns=["Present", "Absent"], fill_value=0)
    )
    stage_signal = stage_signal.loc[(stage_signal.sum(axis=1) > 0), :]
    stage_signal.to_csv(OUTPUT_DIR / "rq2_stage_x_preliminary_signal_q1_v2.csv")
    df.to_csv(OUTPUT_DIR / "rq2_tripartite_dataset_q1_v2.csv", index=False)
    positive_df.to_csv(OUTPUT_DIR / "rq2_preliminary_signals_for_manual_review_q1_v2.csv", index=False)
    denominator_rows = [
        {"rq": "RQ2", "subset": "all_rows", "group": "Evaluated studies", "denominator_n": int(len(df)), "notes": "Analytical universe: graph-ready studies evaluated for RQ2."},
    ]
    for state, count in df["q2_signal_state"].value_counts(dropna=False).reindex(["Present", "Absent", "Uncoded"], fill_value=0).items():
        denominator_rows.append(
            {
                "rq": "RQ2",
                "subset": "title_abstract_signal_state",
                "group": state,
                "denominator_n": int(count),
                "notes": "Preliminary title/abstract evidence only; not confirmed clinical integration.",
            }
        )
    pd.DataFrame(denominator_rows).to_csv(OUTPUT_DIR / "rq2_denominators_q1_v2.csv", index=False)
    fig, heat_ax = plt.subplots(figsize=(8.6, 6.5), facecolor="white")
    heat_values = stage_signal.to_numpy(dtype=float)
    img = heat_ax.imshow(heat_values, cmap="Blues", aspect="auto")
    # Determine threshold for switching annotation colour (white on dark, black on light)
    vmin, vmax = heat_values.min(), heat_values.max()
    threshold = vmin + (vmax - vmin) * 0.55
    row_totals = heat_values.sum(axis=1)
    for row_idx in range(heat_values.shape[0]):
        for col_idx in range(heat_values.shape[1]):
            value = int(heat_values[row_idx, col_idx])
            share = 0.0 if row_totals[row_idx] == 0 else value / row_totals[row_idx]
            if value == 0:
                label = "0"
                font_color = "#888888"
            else:
                label = f"{value}\n{share:.0%}"
                font_color = "white" if heat_values[row_idx, col_idx] >= threshold else "#1a1a2e"
            heat_ax.text(
                col_idx, row_idx, label,
                ha="center", va="center",
                fontsize=10.4, fontweight="bold",
                color=font_color,
                linespacing=1.4,
            )
    heat_ax.set_xticks(range(len(stage_signal.columns)))
    heat_ax.set_xticklabels(["Preliminary signal", "Absent"], fontsize=12)
    heat_ax.set_yticks(range(len(stage_signal.index)))
    heat_ax.set_yticklabels(stage_signal.index, fontsize=11.5)
    heat_ax.set_xlabel("Title/abstract signal state", fontsize=14.5, fontweight="bold", labelpad=14)
    heat_ax.set_ylabel("Clinical stage", fontsize=14, fontweight="bold", labelpad=12)
    # Add a colourbar for scale reference
    cbar = fig.colorbar(img, ax=heat_ax, shrink=0.7, pad=0.02)
    cbar.set_label("Study count", fontsize=11.5)
    cbar.ax.tick_params(labelsize=10.5)
    fig.tight_layout()
    save_figure_variants(fig, OUTPUT_DIR / "rq2_heatmap_and_timing_q1_v2")
    write_caption(
        OUTPUT_DIR / "rq2_heatmap_and_timing_q1_v2_caption.txt",
        """
        RQ2. Preliminary title/abstract evidence heatmap. The heatmap shows 242 studies with preliminary signal and
        34 without signal, stratified only by primary clinical stage. These labels are screening-level evidence
        only and must not be reported as confirmed clinical integration. Cell annotations report the count and the
        within-stage percentage. The 242 signal-positive records are exported for manual review in
        rq2_preliminary_signals_for_manual_review_q1_v2.csv. Study-level classifications are stored in
        rq2_tripartite_dataset_q1_v2.csv and denominators are listed in
        rq2_denominators_q1_v2.csv.
        """,
    )


def _self_check() -> None:
    assert derive_q2_signal_state(pd.Series({"q2_candidate_abstract": "true", "q2_candidate_terms": pd.NA})) == "Present"


if __name__ == "__main__":
    _self_check()
    main()
