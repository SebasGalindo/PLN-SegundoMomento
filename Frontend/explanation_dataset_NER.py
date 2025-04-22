import streamlit as st

st.set_page_config(page_title="Explicación: create_NER_dataset.py", layout="wide")

st.title("Análisis del Script: `create_NER_dataset.py` (Generación de Datos NER)")

st.header("Propósito General y Rol en el Chatbot")

st.markdown("""
Este script se dedica a la **generación sintética de datos específicamente diseñados para entrenar el modelo de Reconocimiento de Entidades Nombradas (NER)**. A diferencia de `create_responses_dataset.py` (que generaba frases para clasificar la intención general), `create_NER_dataset.py` se enfoca exclusivamente en crear ejemplos de frases donde **aparecen las entidades de interés** (como nombres de empresas, valores monetarios, números de empleados, etc.) y, de manera crucial, **anota la posición exacta (índices de inicio y fin de caracteres) de cada entidad dentro de la frase generada**.

**Rol en el Chatbot:** El resultado de este script es el archivo `dataset_ner.jsonl`. Este archivo es el **conjunto de entrenamiento fundamental** para la segunda parte del notebook `BETO_Clasificacion_Frases.ipynb` (o `PLN2Momento (1).ipynb`), donde se realiza el fine-tuning del modelo `AutoModelForTokenClassification`. Ese modelo NER afinado, guardado como `modelo_ner_frases`, es el que permite a la función `predict_ner` del chatbot extraer la información específica proporcionada por el usuario durante la conversación (ej., identificar "500 millones" como `VALOR_ACTIVOS`). La precisión del modelo NER depende directamente de la calidad y la correcta anotación de las entidades en el dataset generado por este script.
""")

st.header("Desglose Técnico del Código")

st.subheader("1. Importaciones y Configuración")

st.markdown("""
Se importan librerías similares a las del generador de datos de intención: `random`, `pandas`, `Faker`, `num2words`, `os`, `numpy`, y adicionalmente `json` para manejar el formato de salida.

La configuración establece:
* `NUM_EJEMPLOS_POR_CATEGORIA_RESPUESTA`: Cuántos ejemplos generar *por cada tipo de respuesta que contenga una entidad*. El foco está en las respuestas, no en todas las intenciones.
* `OUTPUT_JSONL_FILE`: El nombre del archivo de salida. Se elige el formato **JSON Lines (`.jsonl`)** porque es un estándar práctico para datasets de secuencia: cada línea del archivo es un objeto JSON independiente, facilitando el procesamiento. Cada objeto contendrá el texto y las anotaciones de entidades para esa frase.
* `CATEGORIAS_RESPUESTA_NER`: Una lista específica que incluye *solo* aquellas categorías de intención (definidas en el script anterior) que se espera contengan las entidades a extraer (ej: "Respuesta Nombre", "Respuesta Activos").
* `ENTITY_LABELS`: Un diccionario **crucial** que mapea cada categoría de respuesta relevante a la **etiqueta NER específica** que debe asociarse a la información generada dentro de esa respuesta (ej: la categoría "Respuesta Empleados" generará texto asociado a la etiqueta NER "NUM_EMPLEADOS").
""")
st.code("""
import random
import pandas as pd
from faker import Faker
from num2words import num2words
import os
import numpy as np
import json # <--- Para formato JSON Lines

# --- Configuración ---
NUM_EJEMPLOS_POR_CATEGORIA_RESPUESTA = 2000
LOCALE = 'es'
OUTPUT_JSONL_FILE = 'Data/dataset_ner.jsonl' # <--- Salida JSONL

FILES = { # Solo necesitamos las categorías/sectores aquí
    'categorias_sectores': 'Data/Categorias-Empresa.txt'
}

# Solo categorías que contendrán entidades anotadas
CATEGORIAS_RESPUESTA_NER = [
    "Respuesta Nombre", "Respuesta Categoria Empresa", "Respuesta Sector",
    "Respuesta Empleados", "Respuesta Ganancias", "Respuesta Activos",
    "Respuesta Cartera", "Respuesta Deudas",
]

# Mapeo de Categoría de Generación -> Etiqueta NER
ENTITY_LABELS = {
    "Respuesta Nombre": "NOMBRE_EMPRESA",
    "Respuesta Categoria Empresa": "CATEGORIA_EMPRESA",
    "Respuesta Sector": "SECTOR",
    "Respuesta Empleados": "NUM_EMPLEADOS",
    "Respuesta Ganancias": "VALOR_GANANCIAS",
    "Respuesta Activos": "VALOR_ACTIVOS",
    "Respuesta Cartera": "VALOR_CARTERA",
    "Respuesta Deudas": "VALOR_DEUDAS",
}

fake = Faker(LOCALE)
""", language="python")

st.subheader("2. Carga de Recursos y Plantillas")

st.markdown("""
Se reutilizan funciones auxiliares como `leer_categorias_sectores_txt` y `numero_a_palabras`. Se cargan los datos de categorías/sectores si están disponibles. Es importante destacar que **se reutilizan las mismas listas de plantillas** definidas en `create_responses_dataset.py` (ej: `plantillas_respuesta_nombre`, `plantillas_respuesta_empleados`, etc.). Esto asegura consistencia, pero la función de generación se adaptará para extraer las entidades de estas plantillas.
""")
st.code("""
def leer_categorias_sectores_txt(ruta_archivo_txt):
    # ... (Código igual al script anterior) ...
    return df_categorias

def numero_a_palabras(numero):
    # ... (Código igual al script anterior) ...
    return palabras_o_numero_formateado

# --- Cargar Datos Necesarios ---
df_categorias_global = leer_categorias_sectores_txt(FILES['categorias_sectores'])

# --- Plantillas de Respuesta ---
# (Se asume que las listas plantillas_respuesta_* están definidas aquí,
#  igual que en create_responses_dataset.py)
plantillas_respuesta_nombre = [...]
plantillas_respuesta_empleados = [...]
# ... etc ...
plantillas_respues_deuda_ninguna = [...] # Importante manejar este caso
moneda_simbolos = ["COP", "USD", "EUR"]
unidades_dinero_texto = ["pesos", ...]
""", language="python")

st.subheader("3. Generación de Frases y Anotación de Entidades (`generar_frase_ner`)")
st.markdown("""
Esta es la función central y su diferencia clave respecto a `generar_frase` del script anterior reside en su **salida**: no solo genera el texto de la frase, sino también una **lista de tuplas que representan las entidades anotadas**. Cada tupla contiene `(índice_inicio_caracter, índice_fin_caracter, etiqueta_NER)`.

El proceso dentro de la función para cada categoría de `CATEGORIAS_RESPUESTA_NER` es:
1.  **Generar Valor de Entidad:** Se genera un valor realista para la entidad correspondiente (un nombre de empresa con `Faker`, un número de empleados aleatorio dentro de rangos razonables, un valor monetario, etc.). Se aplican técnicas para variar el formato (ej: números como dígitos vs. texto, inclusión de símbolos de moneda o unidades textuales).
2.  **Seleccionar Plantilla:** Se elige aleatoriamente una plantilla de respuesta de la lista correspondiente a la categoría (ej: una plantilla de `plantillas_respuesta_activos`).
3.  **Formatear Frase:** Se inserta el valor generado (`valor_entidad_str`) en la plantilla seleccionada para crear la frase final (`texto_frase`).
4.  **Localizar Entidad (¡Crucial!):** Se utiliza `texto_frase.find(valor_entidad_str)` para encontrar la posición inicial (`start_idx`) del valor de la entidad *exactamente como fue insertado* dentro de la frase final. Se calcula la posición final (`end_idx`).
5.  **Crear Anotación:** Si se encuentra la entidad, se crea la tupla `(start_idx, end_idx, NER_LABEL)` usando la etiqueta NER correspondiente obtenida del diccionario `ENTITY_LABELS`.
6.  **Manejar Casos Especiales:** Se maneja el caso de "Respuesta Deudas" donde a veces se genera una frase indicando "sin deudas". En este caso, la función devuelve la frase pero una lista de entidades *vacía*, ya que no hay un valor numérico de deuda que anotar. Se incluyen advertencias si `find()` falla, lo que podría indicar problemas con plantillas que modifican demasiado el valor insertado.
7.  **Retorno:** La función devuelve la tupla `(texto_frase, lista_entidades)`.

**Relevancia para NER/BERT:** Esta función genera los datos *supervisados* que el modelo NER necesita. Proporciona ejemplos de texto (`texto_frase`) junto con la ubicación exacta (`start_idx`, `end_idx`) y el tipo (`NER_LABEL`) de la información que se espera que el modelo aprenda a extraer. El fine-tuning del modelo (`AutoModelForTokenClassification`) utilizará estas anotaciones a nivel de caracter para crear las etiquetas BIO a nivel de token.
""")
st.code("""
def generar_frase_ner(categoria, df_categorias):
    # ... (Acceso a variables globales: plantillas, fake, ENTITY_LABELS, etc.) ...
    texto_frase = ""
    entidades = [] # Lista para [(start, end, label), ...]
    valor_entidad_str = "" # String exacto de la entidad generada

    if categoria == "Respuesta Nombre":
        plantilla = random.choice(plantillas_respuesta_nombre)
        if plantilla == "{nombre_empresa}": plantilla = "La empresa es {nombre_empresa}." # Evitar solo placeholder
        valor_entidad_str = fake.company()
        texto_frase = plantilla.format(nombre_empresa=valor_entidad_str)

    elif categoria == "Respuesta Empleados":
        # ... (Generar número 'numero' basado en sector) ...
        if random.random() < 0.5: # Formato Dígitos
            valor_entidad_str = f"{numero:,}".replace(",", ".")
        else: # Formato Texto
            valor_entidad_str = numero_a_palabras(numero)
        plantilla = random.choice(plantillas_respuesta_empleados)
        if "{valor}" not in plantilla: plantilla = "Somos {valor} empleados."
        texto_frase = plantilla.format(valor=valor_entidad_str)

    elif categoria == "Respuesta Deudas":
         numero = random.randint(0, 60_000_000_000)
         if numero == 0 or random.random() < 0.05: # Probabilidad de no tener deuda
              texto_frase = random.choice(plantillas_respues_deuda_ninguna)
              # ¡Importante! No hay entidad numérica que anotar aquí
              valor_entidad_str = "" # Marcar que no hay entidad string a buscar
              # Se devuelve la frase, pero la lista de entidades estará vacía
         else:
              # ... (Generar valor_entidad_str con formato dígito+moneda o texto+unidad) ...
              valor_entidad_str = "..." # Ej: "1.500.000 COP" o "unos quince millones de pesos"
              plantilla = random.choice(plantillas_respuesta_deudas)
              if "{valor}" not in plantilla: plantilla = "Debemos {valor}."
              texto_frase = plantilla.format(valor=valor_entidad_str)

    # ... (Lógica similar para otras categorías NER: Categoria Empresa, Sector, Ganancias, Activos, Cartera) ...
    #     Generar valor -> Determinar valor_entidad_str -> Elegir plantilla -> Formatear texto_frase
    # --- Fin de la generación ---

    # --- Encontrar índices y añadir a la lista de entidades ---
    if valor_entidad_str and texto_frase: # Solo si se generó una entidad buscable
        start_idx = texto_frase.find(valor_entidad_str)
        if start_idx != -1:
            end_idx = start_idx + len(valor_entidad_str)
            label = ENTITY_LABELS.get(categoria) # Obtener 'NOMBRE_EMPRESA', 'NUM_EMPLEADOS', etc.
            if label:
                entidades.append((start_idx, end_idx, label)) # Guardar como tupla
            else:
                 print(f"ADVERTENCIA: No se encontró etiqueta NER mapeada para la categoría {categoria}")
        else:
            # Advertencia si no se encuentra el string exacto
            print(f"ADVERTENCIA: No se encontró la entidad '{valor_entidad_str}' en la frase '{texto_frase}'")

    # Devolver texto y lista de entidades (puede estar vacía)
    return texto_frase, entidades
""", language="python")


st.subheader("4. Bucle Principal de Generación NER y Guardado en JSON Lines")

st.markdown("""
El código itera sobre la lista `CATEGORIAS_RESPUESTA_NER` (solo las relevantes para NER). Para cada categoría, llama a `generar_frase_ner` repetidamente hasta generar el número deseado de ejemplos (`NUM_EJEMPLOS_POR_CATEGORIA_RESPUESTA`).
* **Unicidad:** Utiliza un `set` (`textos_generados_set`) para asegurar que no se añadan frases con texto idéntico, manteniendo la diversidad del dataset.
* **Formato de Salida:** Cada par `(texto_frase, lista_entidades)` generado y validado se convierte en un diccionario con claves `"text"` y `"entities"`. La clave `"entities"` contiene la lista de tuplas `(start, end, label)`.
* **Guardado (`.jsonl`):** Se abre el archivo `OUTPUT_JSONL_FILE` y cada diccionario se convierte a una cadena JSON y se escribe en una **línea separada** del archivo (`f.write(json_record + '\\n')`). Este formato JSON Lines es el esperado por la función `cargar_datos_jsonl` en el notebook de fine-tuning NER.

**Relevancia para NER/BERT:** Este bucle genera el volumen necesario de datos de entrenamiento, asegurando que cada ejemplo tenga el formato correcto (texto y anotaciones de entidad con índices de caracteres) para ser procesado posteriormente en el pipeline de fine-tuning del modelo NER.
""")
st.code("""
# --- Generación del Dataset NER ---
dataset_ner = []
textos_generados_set = set()

if df_categorias_global is None and any(c in ["Respuesta Categoria Empresa", "Respuesta Sector"] for c in CATEGORIAS_RESPUESTA_NER):
     print("ADVERTENCIA: Faltan datos de categorías, algunas entidades no se generarán.")

for categoria in CATEGORIAS_RESPUESTA_NER:
    # ... (Bucle while para generar NUM_EJEMPLOS_POR_CATEGORIA_RESPUESTA) ...
        try:
            texto_frase, entidades = generar_frase_ner(categoria, df_categorias_global)

            # Validar y comprobar unicidad del TEXTO
            if texto_frase and isinstance(texto_frase, str) and \
               "Fallback:" not in texto_frase and \
               texto_frase not in textos_generados_set:

                textos_generados_set.add(texto_frase)
                # Guardar solo si se encontraron entidades (o ajustar si se quieren ejemplos sin entidades)
                # El formato esperado por el notebook es List[Tuple[int, int, str]]
                if entidades: # Asegurarse que entidades no esté vacía
                     dataset_ner.append({"text": texto_frase, "entities": entidades})
                     # Incrementar contador
                # else: Podría añadirse aquí ejemplos negativos (sin entidades)? Depende del diseño.

        except Exception as e:
            # ... (Manejo de errores) ...
    # ... (Fin del bucle while y advertencias) ...

# --- Guardar a JSON Lines ---
if not dataset_ner:
     print("ERROR: No se generaron datos NER válidos.")
else:
     print(f"Guardando el dataset NER en '{OUTPUT_JSONL_FILE}'...")
     try:
         with open(OUTPUT_JSONL_FILE, 'w', encoding='utf-8') as f:
             for entry in dataset_ner: # entry es un dict {"text": ..., "entities": [(s,e,l),...]}
                 # Convertir la tupla interna a lista para serialización JSON estándar si es necesario
                 # entry_serializable = {"text": entry["text"], "entities": [list(t) for t in entry["entities"]]}
                 # json.dumps funciona bien con tuplas, así que la línea anterior no es estrictamente necesaria
                 json_record = json.dumps(entry, ensure_ascii=False)
                 f.write(json_record + '\\n') # Escribir JSON en una línea + salto de línea
         print("¡Dataset NER guardado exitosamente!")
         # ... (Mostrar info del dataset) ...
     except Exception as e:
         print(f"Error al guardar el archivo JSON Lines: {e}")

""", language="python")


st.header("Conclusión e Impacto en el Chatbot")

st.success("""
El script `create_NER_dataset.py` es el encargado de construir el **material de estudio específico para el modelo de Reconocimiento de Entidades Nombradas (NER)**. Genera frases realistas (usando plantillas variadas, Faker, formatos numéricos diversos) que contienen ejemplos de la información clave que el chatbot necesita extraer. Lo fundamental es que **anota con precisión los índices de inicio y fin de cada entidad** dentro de esas frases. El archivo resultante, `dataset_ner.jsonl`, alimenta directamente el proceso de fine-tuning del modelo `AutoModelForTokenClassification` (la segunda parte del notebook `PLN2Momento (1).ipynb`), permitiendo que este aprenda a localizar y clasificar entidades como nombres de empresa, valores monetarios, etc., en el texto libre del usuario, una capacidad esencial para la funcionalidad del chatbot financiero.
""")
