# Artificial intelligence and autism — 421-report manuscript figures

All chart outputs are stored in **one directory**: [`figures/`](figures/).
It contains Figures 2–8 in PNG, PDF and SVG, alongside CSVs with report-level
allocations, stage counts and reporting frequencies. Figure 1 (PRISMA) is
managed in the manuscript and is intentionally not generated here.

## Data

- [`Supplementary_Data_S1_simplificado_421.xlsx`](Supplementary_Data_S1_simplificado_421.xlsx): 421 bibliographic reports S001–S421, supplied in the publication's three-sheet layout with 20 fields, including current RQ1 families, independent model configurations and RQ2/RQ3 coding.
- [`rq1_algorithm_family_421.csv`](rq1_algorithm_family_421.csv): report-level current RQ1 family assignment (21 categories).

RQ1: 285 reports have an identifiable principal family or computational method;
95 describe multiple models without a unique principal family; 32 report neural
networks without a specified architectural subtype; nine have no supportable
principal family. The last three states are **not** additional algorithm families.

## Rebuild

```bash
python -m pip install -r requirements.txt
python sync_supplement.py
python generate_manuscript_figures.py
python verify_figures_421.py
```

To regenerate Figure 2 alone: `python plot_figure2_reclassified.py`.
To generate a subset: `python generate_manuscript_figures.py --figures 2 3`.
The compatibility entry points for RQ1, RQ2, RQ3 and all figures still use
the same `figures/` directory. No additional results folders are generated.

Figures 2–8 use the prior visual design, with current underlying numbers.
The bubble matrix in Figure 2 displays 21 rows by six functional stages,
with report counts in the circles and a divider before the three
identification states.

The verifier checks unique report IDs and DOI fingerprint, 21 RQ1 category
counts, the matrix total (421), methodological-practice reporting (25 external
validation, 74 multisource integration, 128 explainability, 17 cross-site),
17 selected RQ3 profiles (275 reports), and all exported figure formats.
Neither the figure directory nor the counts represent clinical validation
of the underlying AI systems.

GitHub Actions regenerates the supplement and all figures on changes to
`master`. Python/Matplotlib processing is CPU-only; a GPU is not required.
