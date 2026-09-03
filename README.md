# AI for autism review scripts

Archivo maestro vigente: `cribado_maestro_276_actualizacion_2026-09-02.xlsx`.

Este repositorio genera el dataset analitico y las figuras RQ1-RQ3 desde el archivo maestro como unica fuente de verdad. El corpus base sigue siendo `Base_276` / `RQ1_RQ2_base_276` / `RQ3_base_276`; el dataset derivado solo normaliza tipos, agrega trazabilidad vacia cuando falta y exporta tablas reproducibles.

## Flujo reproducible

```powershell
.\.venv\Scripts\python.exe run_all_q1_v2.py
.\.venv\Scripts\python.exe validate_outputs_q1_v2.py
```

`run_all_q1_v2.py` reconstruye primero `analysis_dataset_q1_v2.xlsx` y despues regenera RQ1, RQ2 y RQ3.

## Politica de conteos

- Corpus analitico final: 276 estudios.
- Candidatos nuevos que pasaron titulo/resumen: 192, incluyendo `bib_index 188`.
- Excluidos nuevos: 174.
- Candidatos pendientes de adjudicacion de texto completo: 191.
- Textos no recuperados: 0.
- Pool candidato provisional: 276 + 192 = 468. No es el N final incluido.
- RQ2 presenta solo 242 senales preliminares de titulo/resumen y 34 ausentes; no es integracion clinica confirmada. Los 242 registros positivos se exportan para revision manual.
- RQ3 muestra 37 perfiles / 262 estudios y omite 10 perfiles sin senales / 14 estudios desde la misma tabla agregada. A nivel individual: 110 estudios con alguna senal y 166 sin senales. La validacion externa es un indicador separado: 15 presentes y 261 ausentes.

## Correcciones auditadas

- `bib_index 188` esta incluido en el maestro solo como candidato provisional. No se infiere decision de texto completo.
- DeepASDPred / `study_id=282` esta codificado en ambas hojas RQ base como `Biological/omics`, `risk-RNA identification` y `Not specified`, con banderas de etapa en 0.
- `q2_candidate_abstract`, `q2_candidate_terms` y las banderas `q3_*_signal` se exportan como enteros `0/1`; `q3_candidate_terms` permanece como texto.
- Las salidas agregan `reviewer_1`, `reviewer_2`, `adjudicator` y `decision_date`; si el maestro no provee datos, quedan vacios.
