from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MASTER_FILE = BASE_DIR / "cribado_maestro_276_actualizacion_2026-09-02.xlsx"
OUTPUT_FILE = BASE_DIR / "analysis_dataset_q1_v2.xlsx"
ANALYTICAL_N = 276
PROVISIONAL_NEW_N = 191
PROVISIONAL_POOL_N = ANALYTICAL_N + PROVISIONAL_NEW_N


def add_paper_order(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if work["study_id"].nunique() != ANALYTICAL_N or len(work) != ANALYTICAL_N:
        raise ValueError(f"Expected {ANALYTICAL_N} unique evaluated studies, found rows={len(work)} unique={work['study_id'].nunique()}.")
    work.insert(0, "paper_order", range(1, len(work) + 1))
    return work


def build_readme() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "item": "Analytical universe",
                "value": ANALYTICAL_N,
                "note": "These are the studies evaluated in the final graph-ready corpus.",
            },
            {
                "item": "Provisional candidate pool",
                "value": PROVISIONAL_POOL_N,
                "note": "276 prior included studies plus 191 new candidates that passed title/abstract screening.",
            },
            {
                "item": "Important distinction",
                "value": "467 is not final included N",
                "note": "The 191 new candidates must be counted in screening/PRISMA flow, but 190 still need full-text adjudication before RQ coding.",
            },
            {
                "item": "Use for PRISMA/paper ordering",
                "value": "paper_order",
                "note": "Sequential order from 1 to 276. Keep study_id as the original traceability identifier.",
            },
            {
                "item": "study_id",
                "value": "Original identifier",
                "note": "study_id can be non-sequential; it is not the evaluated-study count.",
            },
            {
                "item": "Historical/audit workbooks",
                "value": "Not final denominators",
                "note": "Older source and audit sheets may contain historical rows. Use this workbook for final analysis tables.",
            },
            {
                "item": "RQ1/RQ2 source",
                "value": MASTER_FILE.name,
                "note": "Sheet: RQ1_RQ2_base_276.",
            },
            {
                "item": "RQ3 source",
                "value": MASTER_FILE.name,
                "note": "Sheet: RQ3_base_276.",
            },
        ]
    )


def build_prisma_scope() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stage": "Combined provisional candidate pool",
                "n": PROVISIONAL_POOL_N,
                "note": "276 final prior studies plus 191 new candidates that passed screening; not a final included-study denominator.",
            },
            {
                "stage": "New candidates passed title/abstract screening",
                "n": PROVISIONAL_NEW_N,
                "note": "These passed the update screening and must be counted in screening/PRISMA summaries.",
            },
            {
                "stage": "Final analytical corpus evaluated",
                "n": ANALYTICAL_N,
                "note": "Final graph-ready studies evaluated for the review questions in this repository.",
            },
            {
                "stage": "RQ1 evaluated",
                "n": ANALYTICAL_N,
                "note": "Same analytical corpus; no additional exclusion step is applied by RQ1 script.",
            },
            {
                "stage": "RQ2 evaluated",
                "n": ANALYTICAL_N,
                "note": "Same analytical corpus; q2 signal states are derived within these 276 studies.",
            },
            {
                "stage": "RQ3 evaluated",
                "n": ANALYTICAL_N,
                "note": "Same analytical corpus; Q3 practice signals are derived within these 276 studies.",
            },
        ]
    )


def build_included_studies(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "paper_order",
        "study_id",
        "year",
        "title",
        "doi",
        "assigned_to",
        "modalidad",
        "tipo_IA",
        "stage_primary",
    ]
    return df[[col for col in columns if col in df.columns]].copy()


def main() -> None:
    rq12 = add_paper_order(pd.read_excel(MASTER_FILE, sheet_name="RQ1_RQ2_base_276").drop(columns=["paper_order"], errors="ignore"))
    rq3 = add_paper_order(pd.read_excel(MASTER_FILE, sheet_name="RQ3_base_276").drop(columns=["paper_order"], errors="ignore"))
    if rq12["study_id"].tolist() != rq3["study_id"].tolist():
        raise ValueError("RQ1/RQ2 and RQ3 graph-ready study order differs.")
    update = pd.read_excel(MASTER_FILE, sheet_name="Actualizacion_430")
    passed = update[update["eligibility_bucket"].eq("Provisional include")].copy()
    if len(passed) != PROVISIONAL_NEW_N:
        raise ValueError(f"Expected {PROVISIONAL_NEW_N} provisional new candidates, found {len(passed)}.")

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        build_readme().to_excel(writer, sheet_name="README_FINAL", index=False)
        build_prisma_scope().to_excel(writer, sheet_name="PRISMA_scope_467", index=False)
        build_included_studies(rq12).to_excel(writer, sheet_name="included_studies_276", index=False)
        passed.to_excel(writer, sheet_name="new_candidates_passed_191", index=False)
        rq12.to_excel(writer, sheet_name="rq1_rq2_graph_ready", index=False)
        rq3.to_excel(writer, sheet_name="rq3_graph_ready", index=False)

    print(f"Wrote {OUTPUT_FILE.name} with {ANALYTICAL_N} evaluated studies and {PROVISIONAL_POOL_N} provisional candidates counted.")


if __name__ == "__main__":
    main()
