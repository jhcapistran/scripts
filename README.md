# AI for autism review scripts

Archivo maestro vigente: `cribado_maestro_276_actualizacion_FINAL_CORREGIDO_2026-09-07.xlsx`.

Este repositorio genera `analysis_dataset_q1_v2.xlsx` y las figuras RQ1-RQ3 usando exclusivamente ese maestro como fuente de verdad. Los demas XLSX del directorio son legacy y no deben usarse como input.

## Flujo reproducible

```powershell
.\.venv\Scripts\python.exe run_all_q1_v2.py
.\.venv\Scripts\python.exe validate_outputs_q1_v2.py
```

`run_all_q1_v2.py` reconstruye primero el dataset analitico y despues regenera RQ1, RQ2 y RQ3.

## Politica de conteos

Los conteos se derivan del maestro corregido:

- Corpus final: 454 estudios unicos.
- Base historica: 276 estudios desde `RQ1_RQ2_base_276` / `RQ3_base_276`.
- Nuevos incluidos: 178 estudios desde `Extraccion_RQ_nuevos` con `full_text_decision` Include o Include with integrity flag.
- Exclusiones en texto completo: 14 desde `Texto_completo_191`.
- Pendientes de adjudicacion de texto completo: 0.

## Mapeos

Los estudios nuevos se normalizan al esquema existente:

- `data_source_primary` -> `modalidad`
- `ai_type` -> `tipo_IA`
- `ai_algorithm_main` -> `AI_algorithm_main`
- `ai_task_type` -> `AI_task_type`
- `stage_primary` se conserva y alimenta las banderas de etapa.

RQ2 usa solo `rq2_integration_status` y `rq2_role` como campos finales adjudicados. Los campos legacy `q2_candidate_abstract` y `q2_candidate_terms` quedan solo como auditoria historica y no se interpretan como integracion clinica final.

RQ3 usa los campos adjudicados `q3_*`. Para mantener el diseno actual de cuatro practicas, el script resume XAI estricto/parcial como explicabilidad y multisite/cross-site como evidencia entre sitios.

## Limitacion sin inferencia

El maestro no contiene `rq2_integration_status` ni `rq2_role` para los 276 estudios historicos. Esas filas se marcan como `Not adjudicated in master` en RQ2 en vez de inferir integracion desde `q2_candidate_*`.
