from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RQ12_FILE = BASE_DIR / "consolidado_RA_RB_Q3_completado_RQ2_final.xlsx"
RQ3_FILE = BASE_DIR / "RQ3_datos.xlsx"
OUTPUT_FILE = BASE_DIR / "analysis_dataset_q1_v2.xlsx"
ANALYTICAL_N = 276


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
                "value": RQ12_FILE.name,
                "note": "Sheet: Consolidado_por_asignacion.",
            },
            {
                "item": "RQ3 source",
                "value": RQ3_FILE.name,
                "note": "Sheet: RQ3_graph_ready.",
            },
        ]
    )


def build_prisma_scope() -> pd.DataFrame:
    return pd.DataFrame(
        [
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
    rq12 = add_paper_order(pd.read_excel(RQ12_FILE, sheet_name="Consolidado_por_asignacion"))
    rq3 = add_paper_order(pd.read_excel(RQ3_FILE, sheet_name="RQ3_graph_ready"))
    if rq12["study_id"].tolist() != rq3["study_id"].tolist():
        raise ValueError("RQ1/RQ2 and RQ3 graph-ready study order differs.")

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        build_readme().to_excel(writer, sheet_name="README_FINAL", index=False)
        build_prisma_scope().to_excel(writer, sheet_name="PRISMA_scope_276", index=False)
        build_included_studies(rq12).to_excel(writer, sheet_name="included_studies_276", index=False)
        rq12.to_excel(writer, sheet_name="rq1_rq2_graph_ready", index=False)
        rq3.to_excel(writer, sheet_name="rq3_graph_ready", index=False)

    print(f"Wrote {OUTPUT_FILE.name} with {ANALYTICAL_N} evaluated studies.")


if __name__ == "__main__":
    main()
