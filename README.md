# AI–ASD scoping review: figures for the 421-report manuscript

**Authoritative data:** [`Supplementary_Data_S1_simplificado_421.xlsx`](Supplementary_Data_S1_simplificado_421.xlsx), sheet `02_Included_reports` (421 unique reports, S001–S421).

**Exact figures inserted in the revised Word manuscript:** [`figures_manuscript_421/`](figures_manuscript_421/) contains **Figures 2–8**, extracted from `Manuscrito_IA_TEA_actualizado_421_Mendeley.docx`. PNG artwork is byte-for-byte identical to the embedded Word images for Figures 2, 3, 5, 6, 7 and 8. For Figure 4 the exact embedded vector (SVG), and a PDF rendered from it, are supplied; its PNG is intentionally omitted to avoid accepting a byte-mismatched transfer. PDFs preserve the same artwork (vector-based where SVG was available). The manuscript's **Figure 1 / PRISMA is not reproduced here**: its historical search counts still require reconciliation.

The outdated RQ1–RQ3 result folders and figures from the 428-report version were removed. The former 428-record master is preserved in [`archive/`](archive/) for traceability only; it is **not** an input to the new figures.

## Recalculate from the study-level dataset

```bash
python -m pip install -r requirements.txt
python generate_manuscript_figures.py --check-only
python generate_manuscript_figures.py
python verify_figures_421.py
```

The generator reads the 19-field included-reports charting table, checks that the corpus has 421 unique DOIs and exactly the expected, distinct RQ2 maturity categories (307 research only, 95 proposed only, 18 evaluated AI use, 1 not ascertainable), and calculates the table/plot data for Figures 2–8 and Table 3.

Freshly generated SVG, PNG, PDF and the underlying count CSVs are written to [`figures_recomputed_421/`](figures_recomputed_421/) (ignored by Git). The **frozen manuscript artwork is never overwritten** by routine regeneration. The recomputed charts use the same categories, counts, ordering, and general visual language; they need not be pixel-identical to the frozen manuscript artwork because font metrics and rendering dependencies can vary. This separation is intentional: one directory holds exact submission artwork, and the script independently reconstructs its descriptive data.

The original command remains available as a compatibility entry point:

```bash
python run_all_q1_v2.py
```

The `rq1_script.py`, `rq2_script.py`, and `rq3_script.py` entry points regenerate their corresponding pairs/trio of figures from the **same 421-report input**, not from the obsolete 428-record master.

## Figures in the manuscript

| Manuscript figure | What it shows | File |
|---|---|---|
| 2 | Principal algorithm family × functional stage | `figures_manuscript_421/Figure_02.png` |
| 3 | AI category × primary data source × stage | `figures_manuscript_421/Figure_03.png` |
| 4 | Clinical/care integration maturity by stage | `figures_manuscript_421/Figure_04.png` |
| 5 | Intended versus observed decision timing (95 proposed; 18 evaluated) | `figures_manuscript_421/Figure_05.png` |
| 6 | Reporting of four methodological practices (N=421) | `figures_manuscript_421/Figure_06.png` |
| 7 | Methodological-practice reporting by functional stage | `figures_manuscript_421/Figure_07.png` |
| 8 | Seventeen fully specified AI–data–stage profiles (275 reports) | `figures_manuscript_421/Figure_08.png` |

Files ending `_caption.txt` from earlier result folders were deleted together with their obsolete versions. Current captions and interpretation live in the revised Word manuscript.

## Interpretation and limits

- Unit of analysis: **bibliographic report**, not independent dataset or participant cohort.
- `Not specified`, `Not ascertainable`, and `Not reported in assessed sources` remain distinct; none is silently imputed as absence.
- RQ2's 18 evaluated AI uses are point-of-use evaluation, AI-supported intervention, or AI-enabled care-delivery platform; they are *not* 18 validated clinical deployments.
- The 17 displayed RQ3 profiles represent 275 of 421 reports, selected using **n ≥ 8** for fully specified profiles plus one largest profile for an otherwise unrepresented stage. Every report still contributes to the corpus-wide indicators.
- The historical search and PRISMA counts have not been reconciled by this update; do not use the archived 428-report values for publication.

## Verification

`verify_figures_421.py` independently checks the data denominators, RQ3 indicators, the 17-profile rule, the presence and SHA-256 fingerprints of the exact frozen Word image files. It does not claim to replicate every original screening decision.
