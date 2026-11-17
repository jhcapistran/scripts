"""Synchronize the current 421-report RQ1 allocation in Supplementary Data S1.

Only the 34 adjudicated principal-algorithm descriptions and algorithm-family
labels change. RQ2/RQ3, bibliographic records and functional stages are preserved.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from openpyxl import load_workbook
import pandas as pd

ROOT = Path(__file__).resolve().parent
SUPPLEMENT = ROOT / "Supplementary_Data_S1_simplificado_421.xlsx"
ALLOCATION = ROOT / "rq1_algorithm_family_421.csv"
DOI_SHA = "c0391238635a7c8e8c1a4608b1cb72d8a694b658774e326f0908d71df1055a8d"
UNRESOLVED = "Principal family not resolved"
MULTIPLE = "Multiple models: no unique principal family"
NEURAL = "Neural networks: subtype unresolved"

# Verified method-level descriptions from original articles and cited source evidence.
METHODS = {
"S003": "CatBoost biomarker screening; competing models also evaluated.",
"S009": "Random Forest final classifier, 30 trees, Gini criterion.",
"S010": "Superpower Glass intervention; recognition classifier family undisclosed.",
"S057": "Linear SVM classifier; ICA/connectivity as representation.",
"S061": "Inception-ResNet-v2 facial emotion recognition for ASD support.",
"S072": "Logistic regression ASD classifier on emotion confusion features.",
"S075": "Linear SVM classifier after supervised connectivity estimation.",
"S076": "Canvas Dx gradient-boosted decision-tree classifier from original validation; this study optimized thresholds.",
"S091": "Hierarchical clustering of ASD structural variants; Random Forest explains groups.",
"S100": "Empowered Brain intervention; deployed algorithm architecture not disclosed.",
"S118": "Random Forest selected among feature-based EEG classifiers.",
"S154": "K-means, hierarchical, DBSCAN and HDBSCAN clustering.",
"S176": "Random Forest on motor-interaction features.",
"S177": "Starkids 1.0: proprietary EEG ML algorithms; model architecture undisclosed.",
"S190": "Bisecting K-means on fNIRS sample entropy.",
"S191": "Face/landmark classifiers compared; no unique report-level family.",
"S205": "SVM-RFE/Random Forest gene selection; final nomogram estimator unresolved.",
"S206": "Decision tree selected for derived motor coordination features.",
"S214": "Normative brain deviation modelling and canonical correlation analysis.",
"S234": "SGD-labelled method; loss/estimator unknown, LASSO for feature selection.",
"S239": "Stacked boosted-tree learners with logistic-regression integrator in authors' preprint; journal pipeline not independently verified.",
"S245": "k-nearest neighbors final speech classifier; PCA dimensionality reduction.",
"S249": "Evaluated AI therapy platform; internal algorithm not disclosed.",
"S274": "LOVOT social robot evaluated; internal AI family not disclosed.",
"S280": "Super Learner-based motion adjustment in functional-connectivity analysis.",
"S306": "Logistic regression predictor after AdaBoost biomarker selection.",
"S317": "Nomogram estimator unspecified; LASSO/SVM for upstream gene selection.",
"S340": "Random Forest for exploratory tDCS response prediction; other classifiers compared.",
"S362": "Ridge regression from EEG to ELMo semantics; Euclidean-distance decision.",
"S377": "Doubao conversational platform; underlying model architecture undisclosed.",
"S403": "CEMiTool co-expression and LASSO/SVM-RFE biomarker selection.",
"S404": "Bagging-SVM selected for seven-item M-CHAT-R screening.",
"S407": "SE-CNN features with Random Forest final ASD/TD classification.",
"S421": "Stakeholder evaluation of ML-assisted platform; classifier family undisclosed.",
}

def main():
    mapping = pd.read_csv(ALLOCATION, dtype="string")
    assert mapping.columns.tolist() == ["Report ID", "Algorithm family"]
    ids = [f"S{i:03}" for i in range(1, 422)]
    if mapping["Report ID"].tolist() != ids:
        raise ValueError("Missing or out-of-order included report identifiers")
    assignment = dict(zip(ids, mapping["Algorithm family"].tolist()))
    counts = Counter(assignment.values())
    if len(counts) != 21 or (counts[UNRESOLVED], counts[MULTIPLE], counts[NEURAL]) != (9, 95, 32):
        raise ValueError("Wrong RQ1 category distribution")
    assert len(METHODS) == 34
    wb = load_workbook(SUPPLEMENT)
    ws = wb["02_Included_reports"]
    fields = {ws.cell(2, j).value: j for j in range(1, 20)}
    actual = [ws.cell(r, fields["Report ID"]).value for r in range(3, 424)]
    if ws.max_row != 423 or actual != ids:
        raise ValueError("Workbook must contain S001-S421 exactly once")
    dois = [str(ws.cell(r, fields["DOI"]).value or "").strip().lower().removeprefix("https://doi.org/")
            for r in range(3, 424)]
    fingerprint = hashlib.sha256(("\n".join(dois) + "\n").encode()).hexdigest()
    if fingerprint != DOI_SHA:
        raise ValueError("Report DOI fingerprint does not match the frozen corpus")
    for r, rid in enumerate(ids, 3):
        ws.cell(r, fields["Algorithm family"], assignment[rid])
        if rid in METHODS:
            ws.cell(r, fields["Principal algorithm"], METHODS[rid])
    for sheet in wb:
        for row in sheet:
            for cell in row:
                if getattr(cell, "comment", None) is not None:
                    cell.comment = None
    cb = wb["03_Coding_framework"]
    for r in range(3, cb.max_row + 1):
        field = cb.cell(r, 2).value
        if field == "Publication year":
            cb.cell(r, 5, "Publication year reported in the bibliographic record.")
        if field == "Principal algorithm":
            cb.cell(r, 3, "Principal algorithm or computational method used for the report objective.")
            cb.cell(r, 5, "Feature extraction or selection does not necessarily identify the final model.")
        if field == "Algorithm family":
            cb.cell(r, 3, "One of 21 mutually exclusive current principal-family categories or identification states.")
            cb.cell(r, 4, "; ".join(sorted(counts)))
            cb.cell(r, 5, "One report contributes once; the three identification states are not algorithm families.")
    for r in range(cb.max_row, 2, -1):
        if cb.cell(r, 2).value == "Source-scope limitation":
            cb.delete_rows(r)
    for label, definition in {
        MULTIPLE: "Multiple model families with no unique supportable principal family.",
        NEURAL: "A neural network is documented, but architectural subtype cannot be identified.",
        UNRESOLVED: "AI application or auxiliary algorithms are described, but the principal estimator family cannot be established.",
    }.items():
        if not any(cb.cell(r, 2).value == label for r in range(3, cb.max_row + 1)):
            cb.append(["RQ1", label, definition, label, "This state remains in the denominator."])
    if "04_Current_RQ1_counts" in wb.sheetnames:
        del wb["04_Current_RQ1_counts"]
    summary = wb.create_sheet("04_Current_RQ1_counts")
    stages = ["Prescreening", "Screening", "Diagnosis", "Prognosis", "Monitoring/intervention", "Not specified"]
    summary.append(["Current algorithm-family classification", "Reports", "Share", *stages])
    for family in sorted(counts):
        values = [sum(ws.cell(r, fields["Algorithm family"]).value == family
                      and ws.cell(r, fields["Primary functional stage"]).value == stage
                      for r in range(3, 424)) for stage in stages]
        summary.append([family, counts[family], counts[family] / 421, *values])
    summary.append(["TOTAL", 421, 1.0, *[
        sum(ws.cell(r, fields["Primary functional stage"]).value == stage for r in range(3, 424))
        for stage in stages]])
    summary.column_dimensions["A"].width = 50
    for name in "BCDEFGHI":
        summary.column_dimensions[name].width = 23
    summary.freeze_panes = "B2"
    summary.auto_filter.ref = f"A1:I{summary.max_row}"
    for r in range(2, summary.max_row + 1):
        summary.cell(r, 3).number_format = "0.0%"
    wb.properties.creator = "Research team"
    wb.properties.lastModifiedBy = "Research team"
    wb.properties.title = "Supplementary Data S1: 421 included ASD AI reports"
    wb.properties.subject = "Current study-level codings for RQ1-RQ3"
    wb.properties.description = "Current 421-report inventory; no adjudication history."
    wb.save(SUPPLEMENT)
    print("S1 synchronized:", sum(counts.values()), "reports;", len(counts), "families/states")

if __name__ == "__main__":
    main()
