from __future__ import annotations

from pathlib import Path
import re
import sys
import pandas as pd
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
MASTER_FILE = BASE_DIR / "Maestro_IA_TEA_cierre_2026-09-08.xlsx"

LEGACY_PATTERNS = [
    "cribado_maestro",
    "analysis_dataset_q1_v2.xlsx",
    "RQ3_datos.xlsx",
    "consolidado_RA",
]

RQ2_ADJUDICATED_COLS = [
    "rq2_status",
    "rq2_role",
    "decision_timing_coded",
    "observed_decision_timing",
    "source_level",
    "rationale",
]
RQ2_MATURITY_ORDER = ["Research only", "Proposed only", "Evaluated AI use"]
RQ2_TIMING_ORDER = ["Pre-decision", "In-decision", "Post-decision"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def rq2_maturity(status: object) -> str:
    status_text = str(status)
    if status_text.startswith("Evaluated"):
        return "Evaluated AI use"
    return status_text


def require_frame_equal(actual: pd.DataFrame, expected: pd.DataFrame, message: str) -> None:
    try:
        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
            check_like=False,
            check_names=False,
        )
    except AssertionError as exc:
        raise AssertionError(f"{message}: {exc}") from exc


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


def _sorted_frame(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    df = df.copy()
    for column in df.select_dtypes(include="category"):
        df[column] = df[column].astype(str)
    return df.sort_values(keys).reset_index(drop=True)


def require_table(actual: pd.DataFrame, expected: pd.DataFrame, name: str, keys: list[str]) -> None:
    actual = _sorted_frame(actual, keys)
    expected = _sorted_frame(expected, keys)
    try:
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_names=False, rtol=1e-12, atol=1e-12)
    except AssertionError as exc:
        if actual.shape == expected.shape and list(actual.columns) == list(expected.columns):
            affected = actual.compare(expected, keep_shape=False, keep_equal=False).head(25)
        else:
            affected = pd.concat({"actual": actual.head(12), "expected": expected.head(12)}, names=["source"])
        raise AssertionError(f"{name} does not match BASE_CIERRE.\nAffected records:\n{affected.to_string()}\n{exc}") from exc


def read_summary_sheet(name: str) -> pd.DataFrame:
    return pd.read_excel(MASTER_FILE, sheet_name=name, header=3).dropna(how="all")


def grouped_counts(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df.groupby(columns, dropna=False, sort=True).size().reset_index(name="n")


def with_stage_denominators(counts: pd.DataFrame, inc: pd.DataFrame) -> pd.DataFrame:
    counts = counts.copy()
    counts["denominator_stage"] = counts["stage_primary"].map(inc["stage_primary"].value_counts()).astype(int)
    counts["within_stage_fraction"] = counts["n"] / counts["denominator_stage"]
    return counts


def expected_practices(inc: pd.DataFrame, fields: dict[str, str], profiles: bool) -> pd.DataFrame:
    rows = []
    groups = inc.groupby(["stage_primary", "data_source_primary", "ai_type"], dropna=False, sort=True) if profiles else [("All included", inc), *inc.groupby("stage_primary", dropna=False, sort=False)]
    for key, group in groups:
        for field, label in fields.items():
            values = group[field].astype(str).str.strip()
            denominator = len(group)
            reported = int(values.eq("Reported").sum())
            unknown = int(values.eq("Not ascertainable").sum())
            row = {
                "practice": label,
                "field": field,
                "reported": reported,
                "not_reported": int(values.eq("Not reported in assessed sources").sum()),
                "not_ascertainable": unknown,
                "denominator": denominator,
                "reported_fraction": reported / denominator,
                "fraction_excluding_unknown": reported / (denominator - unknown) if denominator != unknown else 0.0,
            }
            if profiles:
                row = {
                    "stage_primary": key[0],
                    "data_source_primary": key[1],
                    "ai_type": key[2],
                    **row,
                    "study_ids": "; ".join(group["study_id"].astype(str)),
                }
            else:
                row = {"stratum": str(key), **row}
            rows.append(row)
    return pd.DataFrame(rows)


def validate_base_dynamic(base_df: pd.DataFrame, inc: pd.DataFrame) -> None:
    invalid_flags = base_df[~base_df["include_main"].isin([0, 1])]
    require(invalid_flags.empty, "include_main has values other than 0 or 1:\n" + invalid_flags[["study_id", "include_main"]].to_string(index=False))
    require(len(inc) == 428, f"Expected 428 included studies, found {len(inc)}")
    duplicates = inc[inc["study_id"].duplicated(keep=False)][["study_id", "title"]]
    require(inc["study_id"].nunique() == 428, f"Expected 428 unique study_id values, found {inc['study_id'].nunique()}:\n{duplicates.to_string(index=False)}")
    practice_fields = ["external_validation", "multisource_integration", "xai_broad", "cross_site_robustness", "xai_strict", "multisite_dataset", "prospective_evaluation"]
    allowed = {"Reported", "Not reported in assessed sources", "Not ascertainable"}
    for field in practice_fields:
        bad = inc[~inc[field].astype(str).str.strip().isin(allowed)][["study_id", field]]
        require(bad.empty, f"Unexpected category in {field}:\n{bad.to_string(index=False)}")


def validate_excel_dynamic(inc: pd.DataFrame) -> None:
    expected = with_stage_denominators(grouped_counts(inc, ["stage_primary", "data_source_primary", "ai_type"]), inc)
    require_table(read_summary_sheet("RQ1_PERFILES"), expected, "RQ1_PERFILES", ["stage_primary", "data_source_primary", "ai_type"])
    expected = with_stage_denominators(grouped_counts(inc, ["stage_primary", "algorithm_family"]), inc)
    require_table(read_summary_sheet("RQ1_ALGORITMOS"), expected, "RQ1_ALGORITMOS", ["stage_primary", "algorithm_family"])
    expected = with_stage_denominators(grouped_counts(inc, ["stage_primary", "data_source_primary", "algorithm_family", "ai_type"]), inc)
    require_table(read_summary_sheet("RQ1_ALG_FUENTE"), expected, "RQ1_ALG_FUENTE", ["stage_primary", "data_source_primary", "algorithm_family", "ai_type"])
    expected = with_stage_denominators(grouped_counts(inc, ["stage_primary", "rq2_role", "rq2_status", "decision_timing_coded"]), inc)
    require_table(read_summary_sheet("RQ2_INTEGRACION"), expected, "RQ2_INTEGRACION", ["stage_primary", "rq2_role", "rq2_status", "decision_timing_coded"])
    evaluated_sheet = read_summary_sheet("RQ2_USO_EVALUADO")
    evaluated = inc[inc["rq2_status"].astype(str).str.startswith("Evaluated")][evaluated_sheet.columns]
    require_table(evaluated_sheet, evaluated, "RQ2_USO_EVALUADO", ["study_id"])
    all_practices = {
        "external_validation": "External validation",
        "multisource_integration": "Multisource integration",
        "xai_broad": "Explicit explainability (broad)",
        "cross_site_robustness": "Cross-site evaluation",
        "xai_strict": "Formal attribution/explanation (strict)",
        "multisite_dataset": "Multisite data availability",
        "prospective_evaluation": "Prospective AI evaluation",
    }
    require_table(read_summary_sheet("RQ3_PRACTICAS"), expected_practices(inc, all_practices, False), "RQ3_PRACTICAS", ["field", "stratum"])
    primary = {key: all_practices[key] for key in list(all_practices)[:4]}
    require_table(read_summary_sheet("RQ3_PERFILES"), expected_practices(inc, primary, True), "RQ3_PERFILES", ["stage_primary", "data_source_primary", "ai_type", "field"])


def validate_rq1_dynamic(inc: pd.DataFrame) -> None:
    out = BASE_DIR / "rq1_results_q1_v2"
    method = grouped_counts(inc, ["stage_primary", "data_source_primary", "ai_type"]).rename(columns={"stage_primary": "stage", "data_source_primary": "source_modality_display", "ai_type": "ai_method_display", "n": "count"})
    require_table(pd.read_csv(out / "rq1_counts_source_modality_x_method_by_stage.csv"), method, "RQ1 method CSV", ["stage", "source_modality_display", "ai_method_display"])
    algorithm = grouped_counts(inc, ["stage_primary", "algorithm_family", "data_source_primary"]).rename(columns={"stage_primary": "stage", "algorithm_family": "algorithm_display", "data_source_primary": "source_modality_display", "n": "count"})
    require_table(pd.read_csv(out / "rq1_counts_algorithm_x_source_modality_by_stage.csv"), algorithm, "RQ1 algorithm CSV", ["stage", "algorithm_display", "source_modality_display"])
    audit_cols = ["study_id", "cohort", "title", "year", "doi", "data_source_primary", "ai_type", "algorithm_main", "algorithm_family", "stage_primary"]
    expected_audit = inc[audit_cols].copy()
    for column in ["data_source_primary", "ai_type", "algorithm_main", "algorithm_family", "stage_primary"]:
        expected_audit[column] = expected_audit[column].astype(str)
    require_table(pd.read_csv(out / "rq1_stage_assignment_audit.csv"), expected_audit, "RQ1 row audit CSV", ["study_id"])
    summary = inc["stage_primary"].astype(str).value_counts().rename_axis("stage_primary").reset_index(name="studies_n")
    require_table(pd.read_csv(out / "rq1_stage_assignment_summary.csv"), summary, "RQ1 stage summary CSV", ["stage_primary"])
    mapping = pd.DataFrame({"algorithm_raw": inc["algorithm_main"].astype(str), "algorithm_plot": inc["algorithm_family"].astype(str)}).drop_duplicates().sort_values(["algorithm_plot", "algorithm_raw"])
    require_table(pd.read_csv(out / "rq1_algorithm_plot_mapping.csv"), mapping, "RQ1 algorithm mapping CSV", ["algorithm_plot", "algorithm_raw"])
    notes_rows = []
    for _, row in inc.iterrows():
        notes = []
        if pd.notna(row.get("rationale")):
            notes.append(f"Rationale: {row['rationale']}")
        if pd.notna(row.get("source_scope_caution_detail")):
            notes.append(f"Scope caution: {row['source_scope_caution_detail']}")
        if pd.notna(row.get("integrity_caution")):
            notes.append(f"Integrity caution: {row['integrity_caution']}")
        if notes:
            notes_rows.append({"study_id": row["study_id"], "cohort": row.get("cohort", ""), "title": row["title"], "notes_combined": " | ".join(notes)})
    require_table(pd.read_csv(BASE_DIR / "rq1_supporting_q1_v2" / "rq1_notes_full_context.csv"), pd.DataFrame(notes_rows), "RQ1 notes CSV", ["study_id"])
    denom = pd.read_csv(out / "rq1_denominators_q1_v2.csv").set_index("group")["denominator_n"].to_dict()
    expected_denom = {"Evaluated studies": len(inc), "Stage resolved": int(inc["stage_primary"].ne("Not specified").sum()), "Not specified": int(inc["stage_primary"].eq("Not specified").sum())}
    require(denom == expected_denom, f"RQ1 denominator mismatch: actual={denom}; expected={expected_denom}")


def validate_rq2_dynamic(inc: pd.DataFrame) -> None:
    out = BASE_DIR / "rq2_results_q1_v2"
    work = inc.copy()
    work["integration_maturity"] = work["rq2_status"].map(rq2_maturity)
    maturity_counts = work["integration_maturity"].value_counts().reindex(RQ2_MATURITY_ORDER, fill_value=0)
    require(maturity_counts.to_dict() == {"Research only": 311, "Proposed only": 98, "Evaluated AI use": 19}, f"RQ2 maturity changed in BASE_CIERRE: {maturity_counts.to_dict()}")
    stage_order = ["Prescreening", "Screening", "Diagnosis", "Prognosis", "Monitoring/intervention", "Not specified"]
    maturity_table = pd.crosstab(work["stage_primary"], work["integration_maturity"]).reindex(index=stage_order, columns=RQ2_MATURITY_ORDER, fill_value=0).rename_axis("stage_norm").reset_index()
    require_table(pd.read_csv(out / "rq2_stage_x_integration_maturity_q1_v2.csv"), maturity_table, "RQ2 maturity CSV", ["stage_norm"])
    status_order = ["Evaluated at point of use", "Evaluated AI-supported intervention", "Evaluated care-delivery platform", "Proposed only", "Research only"]
    status_table = pd.crosstab(work["stage_primary"], work["rq2_status"]).reindex(index=stage_order, columns=status_order, fill_value=0).rename_axis("stage_norm").reset_index()
    require_table(pd.read_csv(out / "rq2_stage_x_integration_status_q1_v2.csv"), status_table, "RQ2 status CSV", ["stage_norm"])
    proposed = work[work["integration_maturity"].isin(["Proposed only", "Evaluated AI use"])]
    evaluated = work[work["integration_maturity"].eq("Evaluated AI use")]
    specs = [
        ("rq2_role_proposed_evaluated_q1_v2.csv", "rq2_role", proposed, sorted(proposed["rq2_role"].dropna().unique())),
        ("rq2_decision_timing_coded_proposed_evaluated_q1_v2.csv", "decision_timing_coded", proposed, RQ2_TIMING_ORDER),
        ("rq2_observed_decision_timing_evaluated_q1_v2.csv", "observed_decision_timing", evaluated, RQ2_TIMING_ORDER),
    ]
    for filename, column, subset, order in specs:
        expected = subset[column].value_counts().reindex(order, fill_value=0).rename_axis(column).reset_index(name="n")
        require_table(pd.read_csv(out / filename), expected, filename, [column])
    require_frame_equal(pd.read_csv(out / "rq2_adjudicated_columns_q1_v2.csv"), work[RQ2_ADJUDICATED_COLS], "RQ2 adjudicated CSV mismatch")
    trip_cols = ["study_id", "cohort", "title", "doi", "year", "stage_primary", *RQ2_ADJUDICATED_COLS]
    require_table(pd.read_csv(out / "rq2_tripartite_dataset_q1_v2.csv"), work[trip_cols], "RQ2 tripartite CSV", ["study_id"])
    bad = work[~work["integration_maturity"].eq("Evaluated AI use") & ~work["observed_decision_timing"].eq("Not observed in assessed sources")]
    require(bad.empty, "Non-evaluated studies have observed timing:\n" + bad[["study_id", "rq2_status", "observed_decision_timing"]].to_string(index=False))
    denominator_actual = pd.read_csv(out / "rq2_denominators_q1_v2.csv").set_index(["subset", "group"])["denominator_n"].to_dict()
    denominator_expected = {
        ("all_rows", "All included"): len(work),
        ("integration_maturity", "Research only"): int(maturity_counts["Research only"]),
        ("integration_maturity", "Proposed only"): int(maturity_counts["Proposed only"]),
        ("integration_maturity", "Evaluated AI use"): len(evaluated),
        ("role_and_coded_timing", "Proposed/Evaluated"): len(proposed),
        ("observed_timing", "Evaluated AI use"): len(evaluated),
        **{("adjudicated_status", status): int(work["rq2_status"].eq(status).sum()) for status in status_order},
    }
    require(denominator_actual == denominator_expected, f"RQ2 denominator CSV mismatch: actual={denominator_actual}; expected={denominator_expected}")


def validate_rq3_dynamic(inc: pd.DataFrame) -> None:
    from rq3_script import build_combo_summary, build_plot_summary

    out = BASE_DIR / "rq3_results_q1_v2"
    fields = ["external_validation", "multisource_integration", "xai_broad", "cross_site_robustness"]
    expected_combo = build_combo_summary(inc)
    combo = pd.read_csv(out / "supplement" / "rq3_combo_summary_q1_v2.csv")
    require_table(combo, expected_combo[combo.columns], "RQ3 raw profile CSV", ["stage_norm", "modality_norm", "method_norm"])
    expected_plot = build_plot_summary(expected_combo)
    plot = pd.read_csv(out / "rq3_lollipop_combos_q1_v2.csv")
    require_table(plot, expected_plot[plot.columns], "RQ3 plotted profile CSV", ["plot_stage", "plot_modality", "plot_method", "specification_group"])
    require(int(plot["combo_n"].sum()) == len(inc), f"RQ3 figure profiles cover {plot['combo_n'].sum()} studies, expected {len(inc)}")
    obsolete = out / "rq3_omitted_zero_profiles_q1_v2.csv"
    require(not obsolete.exists(), f"Obsolete output still exists: {obsolete.name}")
    labels = {"external_validation": "External validation", "multisource_integration": "Multisource integration", "xai_broad": "Explicit explainability (broad)", "cross_site_robustness": "Cross-site evaluation"}
    global_expected = []
    for field in fields:
        values = inc[field].astype(str).str.strip()
        reported = int(values.eq("Reported").sum())
        global_expected.append({"practice_field": field, "practice": labels[field], "reported_n": reported, "not_reported_n": int(values.eq("Not reported in assessed sources").sum()), "not_ascertainable_n": int(values.eq("Not ascertainable").sum()), "total_n": len(inc), "reported_rate": reported / len(inc)})
        status_cols = [f"{field}_{status}_count" for status in ("reported", "not_reported", "not_ascertainable")]
        affected = plot[plot[status_cols].sum(axis=1).ne(plot["combo_n"])][["plot_label", "combo_n", *status_cols]]
        require(affected.empty, f"RQ3 {field} status counts do not equal profile denominator:\n{affected.to_string(index=False)}")
    require_table(pd.read_csv(out / "rq3_global_practice_summary_q1_v2.csv"), pd.DataFrame(global_expected), "RQ3 global CSV", ["practice_field"])
    denominator_actual = pd.read_csv(out / "rq3_denominators_q1_v2.csv").set_index(["subset", "group"])["denominator_n"].to_dict()
    denominator_expected = {("all_rows", "Evaluated studies"): len(inc)}
    denominator_expected.update({("practice_signal", labels[field]): len(inc) for field in fields})
    denominator_expected.update({
        ("additional_practice_signal", "Formal attribution/explanation (strict)"): len(inc),
        ("additional_practice_signal", "Multisite data availability"): len(inc),
        ("additional_practice_signal", "Prospective AI evaluation"): len(inc),
    })
    require(denominator_actual == denominator_expected, f"RQ3 denominator CSV mismatch: actual={denominator_actual}; expected={denominator_expected}")
    combined = pd.read_csv(BASE_DIR / "rq_denominators_q1_v2.csv")
    analytical = combined[combined["rq"].isin(["RQ1", "RQ2", "RQ3"])].reset_index(drop=True)
    expected_analytical = pd.concat([
        pd.read_csv(BASE_DIR / "rq1_results_q1_v2" / "rq1_denominators_q1_v2.csv"),
        pd.read_csv(BASE_DIR / "rq2_results_q1_v2" / "rq2_denominators_q1_v2.csv"),
        pd.read_csv(out / "rq3_denominators_q1_v2.csv"),
    ], ignore_index=True)
    require_frame_equal(analytical, expected_analytical, "Combined denominator CSV mismatch")


def validate_figures_dynamic() -> None:
    stems = [
        BASE_DIR / "rq1_results_q1_v2" / "rq1_algorithm_bubbles_q1_v2",
        BASE_DIR / "rq1_results_q1_v2" / "rq1_method_source_stage_heatmap_q1_v2",
        BASE_DIR / "rq2_results_q1_v2" / "rq2_integration_levels_and_timing_q1_v2",
        *[BASE_DIR / "rq3_results_q1_v2" / f"rq3_practice_lollipop_{letter}_q1_v2" for letter in "abcd"],
    ]
    for stem in stems:
        for suffix in (".png", ".pdf", ".svg"):
            path = stem.with_suffix(suffix)
            require(path.exists() and path.stat().st_size > 1000, f"Missing or empty figure: {path.name}")
        with Image.open(stem.with_suffix(".png")) as image:
            dpi = image.info.get("dpi", (0, 0))
            require(min(dpi) >= 590, f"{stem.name}.png is not 600 dpi (metadata={dpi})")
    captions = "\n".join(path.read_text(encoding="utf-8") for path in (BASE_DIR / "rq3_results_q1_v2").glob("*caption.txt"))
    require(not re.search(r"\bimplement(?:ed|ation)\b", captions, re.IGNORECASE), "RQ3 captions still use implemented/implementation instead of reported")


def dynamic_main() -> None:
    require(MASTER_FILE.exists(), f"Master file not found: {MASTER_FILE.name}")
    base_df = pd.read_excel(MASTER_FILE, sheet_name="BASE_CIERRE", skiprows=3)
    inc = base_df[base_df["include_main"].eq(1)].copy()
    checks = [
        ("BASE_CIERRE corpus, uniqueness, and categories", lambda: validate_base_dynamic(base_df, inc)),
        ("Excel summary sheets vs BASE_CIERRE", lambda: validate_excel_dynamic(inc)),
        ("RQ1 CSVs vs BASE_CIERRE", lambda: validate_rq1_dynamic(inc)),
        ("RQ2 CSVs vs BASE_CIERRE", lambda: validate_rq2_dynamic(inc)),
        ("RQ3 CSVs and three-state profiles vs BASE_CIERRE", lambda: validate_rq3_dynamic(inc)),
        ("PNG/PDF/SVG exports and terminology", validate_figures_dynamic),
        ("No legacy input references", check_no_legacy_references),
    ]
    failures = 0
    for name, check in checks:
        try:
            check()
        except Exception as exc:
            failures += 1
            print(f"FAIL | {name}\n{exc}\n")
        else:
            print(f"PASS | {name}")
    print(f"\nExecuted {len(checks)} validation groups: {len(checks) - failures} PASS, {failures} FAIL.")
    if failures:
        sys.exit(1)
    print(f"Verified {len(inc)} unique included studies directly against BASE_CIERRE and every analytical CSV.")


if __name__ == "__main__":
    dynamic_main()
