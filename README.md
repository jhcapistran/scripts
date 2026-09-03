# AI for autism review scripts

Archivo maestro vigente: `cribado_maestro_276_actualizacion_2026-09-02.xlsx`.

Este repositorio genera el dataset analitico y las figuras RQ1-RQ3 sin alterar arbitrariamente el corpus base de 276 estudios. El corpus base sigue siendo `Base_276` / `RQ1_RQ2_base_276` / `RQ3_base_276` del archivo maestro. Las correcciones puntuales se aplican en `analysis_dataset_q1_v2.xlsx` para mantener trazabilidad.

## Flujo reproducible

```powershell
.\.venv\Scripts\python.exe run_all_q1_v2.py
.\.venv\Scripts\python.exe validate_outputs_q1_v2.py
```

`run_all_q1_v2.py` reconstruye primero `analysis_dataset_q1_v2.xlsx` y despues regenera RQ1, RQ2 y RQ3.

## Politica de conteos

- Corpus analitico final: 276 estudios.
- Candidatos nuevos que pasaron titulo/resumen: 192, incluyendo `bib_index 188`.
- Pool candidato provisional: 276 + 192 = 468. No es el N final incluido.
- RQ2 se presenta solo como evidencia preliminar de titulo/resumen, no como integracion clinica confirmada.
- RQ3 external validation debe partir como 261 ausentes + 15 presentes = 276.

## Correcciones auditadas

- `bib_index 188` se incluye solo como candidato provisional. No se infiere decision de texto completo.
- DeepASDPred se recodifica en el dataset analitico como `Biological/omics`, `risk-RNA identification` y `Not specified`, sin cambiar el tamano del corpus base.
- Las columnas binarias del dataset analitico se restauran como enteros `0/1`.
- Las salidas agregan columnas de trazabilidad de revisor/fuente/fecha cuando el maestro las provee; si no existen, quedan como `Not recorded`.
