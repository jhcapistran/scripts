"""Validate the current 421-report plots and data used by Figures 2-8."""
from pathlib import Path
from collections import Counter
import json
import hashlib
import pandas as pd

from generate_manuscript_figures import BASE_DIR, SOURCE, PRACTICES, profiles, load_data
from plot_figure2_reclassified import FAMILIES, COUNTS, DOI_SHA256

FIGURES = BASE_DIR / "figures"
EXPECTED_PRACTICES = {
    "External validation": 25,
    "Multisource integration": 74,
    "Explainability (broad)": 128,
    "Cross-site / cross-dataset evaluation": 17,
}

def main():
    df = load_data(SOURCE)
    practice_counts = {label: int((df[col] == "Reported").sum())
                       for col, label in PRACTICES}
    if practice_counts != EXPECTED_PRACTICES:
        raise AssertionError(f"RQ3 totals: {practice_counts}")
    chosen = profiles(df)
    if len(chosen) != 17 or sum(n for _, n in chosen) != 275:
        raise AssertionError("RQ3 figure 8 does not have 17 profiles / 275 reports")
    csv = pd.read_csv(BASE_DIR / "rq1_algorithm_family_421.csv", dtype="string")
    if csv.columns.tolist() != ["Report ID", "Algorithm family"]:
        raise AssertionError("Unexpected classification columns")
    if csv["Report ID"].tolist() != df["Report ID"].tolist():
        raise AssertionError("RQ1 ID mismatch")
    if [int((csv["Algorithm family"] == f).sum()) for f in FAMILIES] != COUNTS:
        raise AssertionError("RQ1 totals changed")
    if not (df["Algorithm family"].astype(str) == csv["Algorithm family"].astype(str)).all():
        raise AssertionError("Supplement contains a non-current algorithm-family label")
    dois = df["DOI"].astype(str).str.lower().str.strip()
    dois = dois.str.replace(r"^https?://(dx\.)?doi\.org/", "", regex=True)
    digest = hashlib.sha256(("\n".join(dois.tolist()) + "\n").encode()).hexdigest()
    if digest != DOI_SHA256:
        raise AssertionError("The DOI inventory differs from the 421-report source")
    table = pd.read_csv(FIGURES / "Figure_02_family_stage_counts.csv")
    if int(table["Reports"].sum()) != 421:
        raise AssertionError("Figure 2 matrix must sum to 421")
    observed = table.set_index(["Algorithm_family", "Primary_functional_stage"])["Reports"]
    expected = df.groupby(["Algorithm family", "Primary functional stage"]).size()
    for (family, stage), n in expected.items():
        if int(observed[(family, stage)]) != int(n):
            raise AssertionError(f"Figure 2 mismatch: {family}, {stage}")
    for n in range(2, 9):
        for ext in ("png", "pdf", "svg"):
            path = FIGURES / f"Figure_{n:02d}.{ext}"
            if not path.is_file() or not path.stat().st_size:
                raise AssertionError(f"Missing current chart: {path}")
    summary = json.loads((FIGURES / "summary_421.json").read_text(encoding="utf-8"))
    if summary["N"] != 421:
        raise AssertionError("The aggregate summary is not for 421 reports")
    print("Verified: 421 reports, RQ1 285/95/32/9, 17 RQ3 profiles / 275 reports")
    print("Reporting totals:", practice_counts)

if __name__ == "__main__":
    main()
