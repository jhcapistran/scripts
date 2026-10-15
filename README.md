# AI for autism review scripts

Archivo maestro único y congelado: `Maestro_IA_TEA_cierre_2026-09-08.xlsx`.

Este repositorio genera las figuras y tablas CSV de RQ1–RQ3 usando exclusivamente este archivo maestro final como única fuente de verdad (leyendo directamente la hoja `BASE_CIERRE` con `include_main == 1`). Los demás archivos XLSX del directorio son legacy y no deben usarse como input.

## Flujo reproducible

```powershell
.\.venv\Scripts\python.exe run_all_q1_v2.py
.\.venv\Scripts\python.exe validate_outputs_q1_v2.py
```

`run_all_q1_v2.py` ejecuta secuencialmente `rq1_script.py`, `rq2_script.py`, `rq3_script.py` y la validación final `validate_outputs_q1_v2.py`.

## Política de conteos y denominadores

Los conteos provienen exclusivamente de `Maestro_IA_TEA_cierre_2026-09-08.xlsx`:

- **Registros conservados para auditoría**: 454 (276 históricos + 178 de actualización).
- **Exclusiones de auditoría**: 26 (23 históricos + 3 de actualización), detalladas en `EXCLUSIONES_CIERRE` e identificadas con `include_main == 0`.
- **Corpus incluido en la síntesis analítica principal**: 428 estudios únicos (253 históricos + 175 de actualización), identificados con `include_main == 1`.
- **RQ1**: 397 resueltos en las 5 etapas clínicas funcionales (Diagnosis: 216, Screening: 113, Monitoring/intervention: 38, Prognosis: 22, Prescreening: 8) y 31 no especificados (*Not specified*). Total = 428.
- **RQ2**: Clasificación adjudicada de madurez de integración (`rq2_status`): Research only: 311, Proposed only: 98, Evaluated AI use: 19 (8 en punto de uso, 10 en intervención, 1 en plataforma).
- Las visualizaciones editoriales de **RQ2** son `rq2_integration_maturity_by_stage_q1_v2` (Figura 4) y `rq2_decision_timing_comparison_q1_v2` (Figura 5), exportadas en PNG, PDF y SVG. La tabla de roles funcionales se elaborará en Word a partir de `rq2_role_proposed_evaluated_q1_v2.csv`; el pipeline no genera una tercera figura ni un archivo de tabla adicional.
- **RQ3**: Cuatro prácticas primarias analizadas en 428 estudios y 90 perfiles originales: External validation: 24 (5.6%), Multisource integration: 77 (18.0%), Explicit explainability broad: 126 (29.4%), Cross-site evaluation: 12 (2.8%). La Figura 6 (`rq3_figure6_practice_reporting_q1_v2`) distingue `Reported`, `Not reported in assessed sources` y `Not ascertainable`; la Figura 7 (`rq3_figure7_stage_practice_heatmap_q1_v2`) compara las seis etapas clínicas; y la Figura 8 (`rq3_figure8_ai_data_stage_matrix_q1_v2`) muestra 17 perfiles completamente especificados seleccionados, que cubren 277 estudios. Los demás estudios permanecen en las Figuras 6–7 y en los CSV analíticos completos. Las tres figuras se exportan en PNG a 600 dpi, PDF y SVG.

La validación reconstruye dinámicamente los cruces, perfiles, categorías, porcentajes y denominadores desde `BASE_CIERRE`, y compara esas reconstrucciones con las hojas resumen del maestro y con cada CSV analítico. Un fallo imprime los registros o celdas afectados y termina con código distinto de cero.

## Sin inferencia de valores faltantes

No se imputa ni infiere ningún valor. Los estados "Not specified", "Not reported in assessed sources" y "Not ascertainable" se respetan explícitamente conforme a la codificación adjudicada del maestro de cierre.
