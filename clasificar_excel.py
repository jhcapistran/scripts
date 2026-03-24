import pandas as pd
import numpy as np

# =========================================
# CONFIGURACIÓN
# =========================================

# Ruta de entrada y salida
EXCEL_PATH = "aceptados_sin_duplicados.xlsx"       # archivo original
SHEET_NAME = "datos_estandarizados"                # hoja con los 837 estudios
OUTPUT_PATH = "aceptados_con_muestra_seleccionada.xlsx"

# Tamaño objetivo de la submuestra
TARGET_N = 300             # número total de artículos a seleccionar
MIN_PER_STRATUM = 2        # mínimo por estrato (si hay suficientes artículos)
RANDOM_SEED = 42           # para reproducibilidad

# Palabras clave clínicas (en inglés y español)
KEYWORDS = [
    # Inglés
    "screening", "diagnosis", "diagnostic", "prognosis",
    "monitoring", "intervention", "treatment",
    "early detection", "early identification", "triage", "clinical",
    # Español (por si hay resúmenes en español)
    "cribado", "diagnóstico", "pronóstico",
    "monitorización", "intervención", "tratamiento",
    "detección temprana", "identificación temprana", "triaje", "clínico"
]

# Pesos para el score (sin año: solo abstract + palabras clave)
W_ABSTRACT = 0.5
W_KEYWORDS = 0.5

# Si quieres que el año forme parte del estrato, pon esto en True.
# Si prefieres que el año no influya *nada* en el muestreo, pon False.
USE_YEAR_BIN_IN_STRATUM = True

# Definición de bins de año si se usan en el estrato
def year_bin(y):
    if pd.isna(y):
        return "unknown"
    y = int(y)
    if y < 2010:
        return "<2010"
    elif y < 2015:
        return "2010-2014"
    elif y < 2020:
        return "2015-2019"
    else:
        return "2020+"

# =========================================
# FUNCIONES AUXILIARES
# =========================================

def count_keywords(text, keywords):
    """Cuenta cuántas palabras clave aparecen en el texto (al menos una vez)."""
    return sum(1 for kw in keywords if kw in text)


# =========================================
# CARGA DE DATOS
# =========================================

df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

# Asegurar que existen columnas mínimas
for required_col in ["title", "modalidad", "tipo_IA"]:
    if required_col not in df.columns:
        raise ValueError(f"Falta la columna obligatoria '{required_col}' en la hoja '{SHEET_NAME}'.")

# Year es útil pero no estrictamente obligatorio
if "year" not in df.columns:
    df["year"] = np.nan
else:
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

# Abstract puede faltar; se trata como cadena vacía
if "abstract" not in df.columns:
    df["abstract"] = ""

# =========================================
# VARIABLES AUXILIARES: ABSTRACT Y TEXTO COMBINADO
# =========================================

df["has_abstract"] = df["abstract"].fillna("").str.strip().ne("")

df["text_combined"] = (
    df["title"].fillna("") + " " + df["abstract"].fillna("")
).str.lower()

df["kw_count"] = df["text_combined"].apply(lambda t: count_keywords(t, KEYWORDS))

# Normalizar conteo de palabras clave
if df["kw_count"].max() > 0:
    df["kw_norm"] = df["kw_count"] / df["kw_count"].max()
else:
    df["kw_norm"] = 0.0

df["has_abstract_int"] = df["has_abstract"].astype(int)

# =========================================
# SCORE DE PRIORIDAD (SIN AÑO)
# =========================================

df["score"] = (
    W_ABSTRACT * df["has_abstract_int"] +
    W_KEYWORDS * df["kw_norm"].fillna(0)
)

# =========================================
# DEFINICIÓN DE ESTRATOS
# =========================================

# Limpiar modalidad y tipo_IA
df["modalidad_str"] = df["modalidad"].fillna("Unknown").astype(str)
df["tipo_IA_str"] = df["tipo_IA"].fillna("Unknown").astype(str)

if USE_YEAR_BIN_IN_STRATUM:
    df["year_bin"] = df["year"].apply(year_bin)
    df["stratum"] = (
        df["modalidad_str"] + " | " + df["tipo_IA_str"] + " | " + df["year_bin"]
    )
else:
    df["stratum"] = (
        df["modalidad_str"] + " | " + df["tipo_IA_str"]
    )

# =========================================
# ASIGNACIÓN DEL TAMAÑO DE MUESTRA POR ESTRATO
# =========================================

group_sizes = df.groupby("stratum").size()
total_n = len(df)

# Si el TARGET_N es mayor que el número total de artículos, lo limitamos
TARGET_N = min(TARGET_N, total_n)

# Asignación proporcional inicial
alloc = ((group_sizes / total_n) * TARGET_N).round().astype(int)

# Ajustar asignaciones mínima/máxima por estrato
for s, n_s in group_sizes.items():
    if n_s == 0:
        alloc[s] = 0
    else:
        alloc_s = alloc.get(s, 0)
        # mínimo: 1 o MIN_PER_STRATUM pero no mayor que el tamaño del estrato
        alloc_s = max(1, alloc_s, MIN_PER_STRATUM)
        alloc[s] = min(alloc_s, n_s)

# Ajustar para que la suma sea aproximadamente TARGET_N
current_total = alloc.sum()

# Reducir si nos pasamos
while current_total > TARGET_N:
    # Estratos con más de 1 artículo asignado
    candidates = alloc[alloc > 1].sort_values(ascending=False)
    if len(candidates) == 0:
        break
    s_to_reduce = candidates.index[0]
    alloc[s_to_reduce] -= 1
    current_total -= 1

# Aumentar si nos quedamos cortos (si es posible)
while current_total < TARGET_N:
    remaining_capacity = group_sizes - alloc
    candidates = remaining_capacity[remaining_capacity > 0].sort_values(ascending=False)
    if len(candidates) == 0:
        break
    s_to_increase = candidates.index[0]
    alloc[s_to_increase] += 1
    current_total += 1

print(f"Total asignado a estratos: {current_total} (objetivo: {TARGET_N})")

# =========================================
# MUESTREO DENTRO DE CADA ESTRATO
# =========================================

rng = np.random.default_rng(RANDOM_SEED)
selected_indices = []

for s, group in df.groupby("stratum"):
    k = alloc.get(s, 0)
    if k <= 0:
        continue

    if len(group) <= k:
        # Si el estrato tiene menos o igual artículos que k, se toman todos
        selected_indices.extend(group.index.tolist())
    else:
        scores = group["score"].fillna(0).to_numpy()
        if scores.sum() <= 0:
            # Si todos los scores son 0, muestreo uniforme
            probs = None
        else:
            probs = scores / scores.sum()
        chosen = rng.choice(group.index.to_numpy(), size=k, replace=False, p=probs)
        selected_indices.extend(chosen.tolist())

# =========================================
# MARCAR SELECCIÓN EN EL DATAFRAME
# =========================================

df["selected_for_full_coding"] = 0
df.loc[selected_indices, "selected_for_full_coding"] = 1

# (Opcional) ordenar para ver primero los seleccionados y con mayor score
df = df.sort_values(
    ["selected_for_full_coding", "score"],
    ascending=[False, False]
)

# =========================================
# GUARDAR RESULTADOS
# =========================================

with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name=SHEET_NAME, index=False)

print(f"Archivo con la muestra seleccionada guardado en: {OUTPUT_PATH}")
print("Número de artículos seleccionados:", df["selected_for_full_coding"].sum())
print("Número total de artículos:", len(df))
