"""Synchronize the current 421-report RQ1 allocation in Supplementary Data S1.

Synchronizes the final 421-report classifications after direct review of 19
full texts, retaining the three-sheet publication layout and all 20 fields.
Only evidence-supported coding fields are revised; no records are added.
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


from copy import copy
from openpyxl.styles import Alignment

MODEL_LABELS = {
    "A": "One principal family/method; full configuration not established",
    "B": "Integrated pipeline or architecture",
    "C": "Independent comparisons or separate tasks",
    "D": "Configuration not established",
    "E": "Prediction ensemble",
    "F": "Integrated pipeline and prediction ensemble",
}
# Current study-level configuration codes from the 421-report supplementary template.
MODEL_CODES = (
    "ACDCCEAEDDDAAACAAAACCCBAACAACCAAACCCDABDBAAACAAAAAAACAAADAAADAAAABBAAAADABBDAABC"
    "BADBBAAAAADACABECAADAAAAAAAABBAAAAAAADBDACACAAAABAAAAAABCCAAABAAABABABBAEDCAAACA"
    "BBEABAAAAAABAAADDAEACAEFABAADDDAACAAAAAAAAABBDAAAAAAADACAAECDAAABAAAADBAADAAAADA"
    "AAAADADBDAECACAAAABAACAAACCCAAAEBDAAAACEEBACACAACCBBAEAADBCEBCAABDABEABBBAABDABB"
    "BBABBBAABAAACBABADBDABABAEAABBAAABAABABAABBAAFAAACDCAEACDCEAEDAABADABADFBAAAACAA"
    "CEDCAABCAAFBCCBAAABAD"
)
METHOD_LABEL_FIXES = {
    "S007": "Shallow feedforward neural network",
    "S012": "Linear SVM",
    "S020": "Random forest selected after comparison with SVM and MLP; two-stage severity classification",
    "S040": "Transformer-family language model proposed; implemented architecture not specified",
    "S058": "Clustering-based analysis; subtype not specified",
    "S081": "Elastic-net classifier; logistic-regression ablation; polynomial gaze regression",
    "S120": "Support vector machines / regression; precise diagnostic-boundary estimator configuration not specified",
    "S137": "Regularized Greedy Forest; ET/RF comparators",
    "S141": "Gradient boosting classifier",
    "S146": "DecisionTreeClassifier on computer-vision gaze measurements",
    "S162": "Supervised canonical graph matching with support-vector classification; graph construction upstream",
    "S164": "SVM for case-control prediction from static/dynamic connectivity; Louvain analysis ancillary",
    "S168": "Random forest on EEG and baseline clinical features",
    "S183": "Convolutional-network classification with atlas-specific majority-vote aggregation; backbone not specified",
    "S194": "Linear SVM with nested principal-component reduction",
    "S216": "XGBoost selected by AU-PRC; RF/DT comparators; k-means on SHAP embeddings for subgroup analysis",
    "S217": "Regression tree with adaptive selection of follow-up questions",
    "S281": "Bagged L2-regularized logistic regression classifiers"
}

# Evidence-supported amendments from the original Methods/Results of 19 PDFs.
# RQ3 labels are mutually exclusive reporting statuses; these edits are idempotent.
EVIDENCE_UPDATES = {
    "S003": {
        "External validation": "Not reported in assessed sources",
        "Explainability: broad": "Reported"
    },
    "S009": {
        "AI category": "Machine Learning",
        "Explainability: broad": "Reported"
    },
    "S057": {
        "Explainability: broad": "Reported"
    },
    "S075": {
        "Explainability: broad": "Reported"
    },
    "S154": {
        "Primary data source": "Multiple source categories",
        "Multisource integration": "Reported"
    },
    "S176": {
        "External validation": "Reported"
    },
    "S190": {
        "Explainability: broad": "Reported"
    },
    "S245": {
        "AI category": "Machine Learning"
    },
    "S306": {
        "Functional role": "Automated measurement/classification",
        "External validation": "Not reported in assessed sources",
        "Explainability: broad": "Reported"
    },
    "S317": {
        "Explainability: broad": "Reported"
    },
    "S362": {
        "Explainability: broad": "Reported"
    }
}

def main():
    mapping = pd.read_csv(ALLOCATION, dtype="string")
    ids = [f"S{i:03d}" for i in range(1, 422)]
    if mapping.columns.tolist() != ["Report ID", "Algorithm family"] or mapping["Report ID"].tolist() != ids:
        raise ValueError("RQ1 allocation must include 421 reports with current columns")
    allocation = dict(zip(ids, mapping["Algorithm family"].tolist()))
    counts = Counter(allocation.values())
    if len(counts) != 21 or (counts[UNRESOLVED], counts[MULTIPLE], counts[NEURAL]) != (9, 95, 32):
        raise ValueError("Inconsistent algorithm-family counts")
    if len(MODEL_CODES) != 421 or set(MODEL_CODES) != set(MODEL_LABELS):
        raise ValueError("Model configuration codes must cover all 421 reports")

    wb = load_workbook(SUPPLEMENT)
    ws = wb["02_Included_reports"]
    fields = {ws.cell(2, col).value: col for col in range(1, 21) if ws.cell(2, col).value is not None}
    family_col = fields.get("Algorithm family / identification status", fields.get("Algorithm family"))
    if family_col != 11:
        raise ValueError("The 11th report field must be the algorithm-family allocation")
    for col, label in ((1, "Report ID"), (6, "DOI"), (10, "Principal algorithm")):
        if fields.get(label) != col:
            raise ValueError(f"Unexpected field position for {label}")

    actual = [ws.cell(row, 1).value for row in range(3, 424)]
    if ws.max_row != 423 or actual != ids:
        raise ValueError("The workbook must include each report S001-S421 once")
    dois = [str(ws.cell(row, 6).value or "").strip().lower().removeprefix("https://doi.org/")
            for row in range(3, 424)]
    fingerprint = hashlib.sha256(("\n".join(dois) + "\n").encode()).hexdigest()
    if fingerprint != DOI_SHA:
        raise ValueError("The DOI inventory does not match the retained corpus")

    # Restore the manuscript's 20-field charting layout; the model-configuration
    # field is independent of the mutually exclusive principal-family allocation.
    ws["A1"] = "Included reports and data charting"
    ws["K2"] = "Algorithm family / identification status"
    ws["T2"] = "Model configuration"
    ws["T2"]._style = copy(ws["S2"]._style)
    ws.column_dimensions["T"].width = 46
    for row, rid in enumerate(ids, 3):
        ws.cell(row, 11, allocation[rid])
        if rid in METHODS:
            ws.cell(row, 10, METHODS[rid])
        if rid in METHOD_LABEL_FIXES:
            ws.cell(row, 10, METHOD_LABEL_FIXES[rid])
        for field, label in EVIDENCE_UPDATES.get(rid, {}).items():
            if field not in fields:
                raise KeyError("Missing supplementary coding field: " + field)
            ws.cell(row, fields[field], label)
        cell = ws.cell(row, 20, MODEL_LABELS[MODEL_CODES[row - 3]])
        cell._style = copy(ws.cell(row, 19)._style)
        cell.alignment = copy(ws.cell(row, 19).alignment)
    expected_practice_counts = {
        "External validation": ("Reported", 26, "Not ascertainable", 5),
        "Multisource integration": ("Reported", 75, "Not ascertainable", 15),
        "Explainability: broad": ("Reported", 136, "Not ascertainable", 8),
        "Cross-site or cross-dataset evaluation": ("Reported", 17, "Not ascertainable", 3),
    }
    for field, (positive, n_positive, uncertain, n_uncertain) in expected_practice_counts.items():
        values = Counter(ws.cell(i, fields[field]).value for i in range(3, 424))
        if values[positive] != n_positive or values[uncertain] != n_uncertain:
            raise ValueError(f"Incorrect current reporting counts for {field}: {values}")
    ai_counts = Counter(ws.cell(i, fields["AI category"]).value for i in range(3, 424))
    source_counts = Counter(ws.cell(i, fields["Primary data source"]).value for i in range(3, 424))
    assert ai_counts["Machine Learning"] == 217 and ai_counts["Not specified"] == 18
    assert source_counts["Multiple source categories"] == 44 and source_counts["Not specified"] == 14
    if "IncludedReports" in ws.tables:
        ws.tables["IncludedReports"].ref = "A2:T423"

    # The coding framework matches the three-sheet publication version.
    cb = wb["03_Coding_framework"]
    rows = [list(r) for r in cb.iter_rows(min_row=3, max_col=5, values_only=True)
            if r[1] is not None]
    for row in rows:
        if row[1] in ("Algorithm family", "Algorithm family / identification status"):
            row[1] = "Algorithm family / identification status"
            row[2] = "One mutually exclusive current principal-family category or identification state per bibliographic report."
            row[3] = "; ".join(sorted(counts))
            row[4] = "Each report contributes once. The three identification states are not additional algorithm families."
        elif row[1] == "Principal algorithm":
            row[2] = "Principal model or computational approach recorded for the report-level objective."
            row[4] = "Auxiliary preprocessing, feature selection and embedding methods are not automatically the final classifier."
        elif row[1] == "Model configuration":
            row[2] = "Study-level arrangement of computational methods, separate from principal algorithm family."
            row[3] = "; ".join(MODEL_LABELS.values())
        elif row[1] == "Publication year":
            row[4] = "Bibliographic year of publication; no temporal trend analysis is reported."
    if not any(r[1] == "Model configuration" for r in rows):
        at = next(i for i, r in enumerate(rows) if r[1] == "Algorithm family / identification status")
        rows.insert(at + 1, ["RQ1", "Model configuration",
                    "Study-level arrangement of computational methods, separate from principal algorithm family.",
                    "; ".join(MODEL_LABELS.values()),
                    "Model comparisons are not necessarily prediction ensembles; sequential pipelines can integrate methods without aggregating predictions."])
    if not any(r[1] == "Source-scope limitation" for r in rows):
        at = next(i for i, r in enumerate(rows) if r[1] == "Search documentation")
        rows.insert(at, ["Interpretation", "Source-scope limitation",
                         "The scope and detail of assessed source material vary across reports.",
                         "Report-level charting and source notes",
                         "Field-level assessments are limited to the available evidence."])
    states = {
        MULTIPLE: "Multiple model families with no unique supportable principal family.",
        NEURAL: "A neural network is identified without an established architectural subtype.",
        UNRESOLVED: "The principal decision-producing estimator family cannot be established.",
    }
    for state, definition in states.items():
        if not any(r[1] == state for r in rows):
            rows.append(["RQ1 identification state", state, definition, state,
                         "The report remains in the denominator; no principal family is inferred."])

    styles = [copy(cb.cell(3, c)._style) for c in range(1, 6)]
    cb.delete_rows(3, cb.max_row - 2)
    for data in rows:
        cb.append(data)
        for c in range(1, 6):
            cell = cb.cell(cb.max_row, c)
            cell._style = copy(styles[c - 1])
            cell.alignment = Alignment(vertical="center", wrap_text=True)
        cb.row_dimensions[cb.max_row].height = 90 if data[1] == "Algorithm family / identification status" else 75
    cb["A1"] = "Coding framework"
    cb.row_dimensions[13].height = 178
    cb.row_dimensions[14].height = 110
    if "CodingFramework" in cb.tables:
        cb.tables["CodingFramework"].ref = f"A2:E{cb.max_row}"

    for sheet in list(wb):
        if sheet.title not in ("01_Search_strategies", "02_Included_reports", "03_Coding_framework"):
            del wb[sheet.title]
    wb["01_Search_strategies"]["A1"] = " Search strategies"
    for sheet in wb:
        for row in sheet:
            for cell in row:
                if getattr(cell, "comment", None) is not None:
                    cell.comment = None

    assert wb.sheetnames == ["01_Search_strategies", "02_Included_reports", "03_Coding_framework"]
    assert cb.max_row == 35 and ws.max_column == 20
    wb.properties.title = "Supplementary Data S1: Search strategies and included-report charting"
    wb.properties.creator = "Authors"
    wb.properties.lastModifiedBy = "Authors"
    wb.properties.subject = "ASD and AI: 421 included reports"
    wb.properties.description = "Current report-level algorithms and model configurations"
    wb.save(SUPPLEMENT)
    print("S1 final: 421 reports, 20 fields, 3 sheets, 16 verified coding corrections")


if __name__ == "__main__":
    main()
