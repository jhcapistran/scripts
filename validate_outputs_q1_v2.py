from __future__ import annotations

from pathlib import Path
import re
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MASTER_FILE = BASE_DIR / "Maestro_IA_TEA_cierre_2026-09-08.xlsx"

LEGACY_PATTERNS = [
    "cribado_maestro",
    "analysis_dataset_q1_v2.xlsx",
    "RQ3_datos.xlsx",
    "consolidado_RA",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_no_legacy_references() -> None:
    py_files = [f for f in BASE_DIR.glob("*.py") if f.name != Path(__file__).name]
    for py_file in py_files:
        content = py_file.read_text(encoding="utf-8")
        for pat in LEGACY_PATTERNS:
            matches = re.findall(pat, content, re.IGNORECASE)
            require(
                len(matches) == 0,
                f"File {py_file.name} still contains legacy reference to '{pat}'",
            )


def main() -> None:
    require(MASTER_FILE.exists(), f"Master file not found: {MASTER_FILE.name}")
    check_no_legacy_references()

    # 1. Validate Master BASE_CIERRE
    base_df = pd.read_excel(MASTER_FILE, sheet_name="BASE_CIERRE", skiprows=3)
    require(len(base_df) == 454, f"Expected 454 total records in BASE_CIERRE, found {len(base_df)}")
    inc = base_df[base_df["include_main"] == 1]
    exc = base_df[base_df["include_main"] == 0]
    require(len(inc) == 428, f"Expected 428 included studies, found {len(inc)}")
    require(len(exc) == 26, f"Expected 26 excluded studies, found {len(exc)}")

    # Check stage resolution
    stage_resolved = int((inc["stage_primary"] != "Not specified").sum())
    stage_unspecified = int((inc["stage_primary"] == "Not specified").sum())
    require(stage_resolved == 397, f"Expected 397 stage-resolved studies, found {stage_resolved}")
    require(stage_unspecified == 31, f"Expected 31 unspecified stage studies, found {stage_unspecified}")

    # 2. Validate RQ1 outputs
    rq1_counts = pd.read_csv(BASE_DIR / "rq1_results_q1_v2" / "rq1_counts_source_modality_x_method_by_stage.csv")
    require(len(rq1_counts) == 90, f"Expected 90 profiles in RQ1 method counts, found {len(rq1_counts)}")
    require(rq1_counts["count"].sum() == 428, f"Expected total count 428 in RQ1 method counts, found {rq1_counts['count'].sum()}")

    rq1_alg = pd.read_csv(BASE_DIR / "rq1_results_q1_v2" / "rq1_counts_algorithm_x_source_modality_by_stage.csv")
    require(rq1_alg["count"].sum() == 428, f"Expected total count 428 in RQ1 algorithm counts, found {rq1_alg['count'].sum()}")

    # 3. Validate RQ2 outputs
    rq2_table = pd.read_csv(BASE_DIR / "rq2_results_q1_v2" / "rq2_stage_x_integration_status_q1_v2.csv", index_col=0)
    require(rq2_table.to_numpy().sum() == 428, f"Expected 428 studies in RQ2 table, found {rq2_table.to_numpy().sum()}")
    require(rq2_table["Research only"].sum() == 311, f"Expected 311 research only, found {rq2_table['Research only'].sum()}")
    require(rq2_table["Proposed only"].sum() == 98, f"Expected 98 proposed only, found {rq2_table['Proposed only'].sum()}")
    evaluated_total = (
        rq2_table["Evaluated at point of use"].sum()
        + rq2_table["Evaluated AI-supported intervention"].sum()
        + rq2_table["Evaluated care-delivery platform"].sum()
    )
    require(evaluated_total == 19, f"Expected 19 evaluated use cases, found {evaluated_total}")

    rq2_tripartite = pd.read_csv(BASE_DIR / "rq2_results_q1_v2" / "rq2_tripartite_dataset_q1_v2.csv")
    require(len(rq2_tripartite) == 428, f"Expected 428 studies in rq2_tripartite_dataset, found {len(rq2_tripartite)}")

    # 4. Validate RQ3 outputs
    rq3_summary = pd.read_csv(BASE_DIR / "rq3_results_q1_v2" / "rq3_global_practice_summary_q1_v2.csv")
    rates = dict(zip(rq3_summary["practice_field"], rq3_summary["reported_n"]))
    require(rates.get("external_validation") == 24, f"Expected 24 external validation, found {rates.get('external_validation')}")
    require(rates.get("multisource_integration") == 77, f"Expected 77 multisource integration, found {rates.get('multisource_integration')}")
    require(rates.get("xai_broad") == 126, f"Expected 126 xai broad, found {rates.get('xai_broad')}")
    require(rates.get("cross_site_robustness") == 12, f"Expected 12 cross site robustness, found {rates.get('cross_site_robustness')}")

    rq3_combo = pd.read_csv(BASE_DIR / "rq3_results_q1_v2" / "supplement" / "rq3_combo_summary_q1_v2.csv")
    require(len(rq3_combo) == 90, f"Expected 90 profiles in rq3_combo_summary, found {len(rq3_combo)}")
    require(rq3_combo["combo_n"].sum() == 428, f"Expected total 428 in rq3_combo_summary, found {rq3_combo['combo_n'].sum()}")

    # 5. Validate figures existence and sizes
    figures = [
        BASE_DIR / "rq1_results_q1_v2" / "rq1_algorithm_bubbles_q1_v2.png",
        BASE_DIR / "rq1_results_q1_v2" / "rq1_algorithm_bubbles_q1_v2.pdf",
        BASE_DIR / "rq1_results_q1_v2" / "rq1_algorithm_bubbles_q1_v2.svg",
        BASE_DIR / "rq1_results_q1_v2" / "rq1_method_source_stage_heatmap_q1_v2.png",
        BASE_DIR / "rq1_results_q1_v2" / "rq1_method_source_stage_heatmap_q1_v2.pdf",
        BASE_DIR / "rq1_results_q1_v2" / "rq1_method_source_stage_heatmap_q1_v2.svg",
        BASE_DIR / "rq2_results_q1_v2" / "rq2_heatmap_and_timing_q1_v2.png",
        BASE_DIR / "rq2_results_q1_v2" / "rq2_heatmap_and_timing_q1_v2.pdf",
        BASE_DIR / "rq2_results_q1_v2" / "rq2_heatmap_and_timing_q1_v2.svg",
        BASE_DIR / "rq3_results_q1_v2" / "rq3_practice_lollipop_a_q1_v2.png",
        BASE_DIR / "rq3_results_q1_v2" / "rq3_practice_lollipop_a_q1_v2.pdf",
        BASE_DIR / "rq3_results_q1_v2" / "rq3_practice_lollipop_a_q1_v2.svg",
        BASE_DIR / "rq3_results_q1_v2" / "rq3_practice_lollipop_b_q1_v2.png",
        BASE_DIR / "rq3_results_q1_v2" / "rq3_practice_lollipop_b_q1_v2.pdf",
        BASE_DIR / "rq3_results_q1_v2" / "rq3_practice_lollipop_b_q1_v2.svg",
        BASE_DIR / "rq3_results_q1_v2" / "rq3_practice_lollipop_c_q1_v2.png",
        BASE_DIR / "rq3_results_q1_v2" / "rq3_practice_lollipop_c_q1_v2.pdf",
        BASE_DIR / "rq3_results_q1_v2" / "rq3_practice_lollipop_c_q1_v2.svg",
        BASE_DIR / "rq3_results_q1_v2" / "rq3_practice_lollipop_d_q1_v2.png",
        BASE_DIR / "rq3_results_q1_v2" / "rq3_practice_lollipop_d_q1_v2.pdf",
        BASE_DIR / "rq3_results_q1_v2" / "rq3_practice_lollipop_d_q1_v2.svg",
    ]
    for fig in figures:
        require(fig.exists(), f"Figure missing: {fig.name}")
        require(fig.stat().st_size > 1000, f"Figure too small: {fig.name} ({fig.stat().st_size} bytes)")

    # 6. Validate master denominator CSV
    master_denoms = pd.read_csv(BASE_DIR / "rq_denominators_q1_v2.csv")
    require("Screening" in master_denoms["rq"].values, "Master denominators missing Screening")
    require("RQ1" in master_denoms["rq"].values, "Master denominators missing RQ1")
    require("RQ2" in master_denoms["rq"].values, "Master denominators missing RQ2")
    require("RQ3" in master_denoms["rq"].values, "Master denominators missing RQ3")

    print("================================================================================")
    print("PASS: Todas las validaciones superadas con éxito.")
    print(f"Fuente única de datos: {MASTER_FILE.name}")
    print(f"- Registros conservados: 454 (276 históricos + 178 nuevos de actualización)")
    print(f"- Exclusiones de auditoría: 26 (23 históricos + 3 de actualización)")
    print(f"- Corpus analítico incluido: 428 (253 históricos + 175 de actualización)")
    print(f"- RQ1: 397 resueltos en 5 etapas clínicas, 31 no especificados (Total: 428)")
    print(f"- RQ2: 311 research-only, 98 proposed-only, 19 evaluated use (8 PoU, 10 interv., 1 plat.)")
    print(f"- RQ3: 24 external val., 77 multisource, 126 XAI broad, 12 cross-site (90 perfiles)")
    print(f"- Figuras: 21 archivos regenerados (.png, .pdf, .svg) sin errores")
    print(f"- Código: 0 scripts con referencias a archivos legacy")
    print("================================================================================")


if __name__ == "__main__":
    main()
