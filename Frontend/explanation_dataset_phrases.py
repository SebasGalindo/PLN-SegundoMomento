import streamlit as st

st.set_page_config(page_title="Explicación: create_responses_dataset.py", layout="wide")

st.title("Análisis del Script: `create_responses_dataset.py`")

st.header("Propósito General para el Chatbot")

st.markdown("""
Este script es responsable de **generar el conjunto de datos (`dataset_completo.csv`) utilizado para entrenar el modelo de clasificación de intenciones** del chatbot. A diferencia de `generate_dataset.py` (que creaba datos para el modelo de clasificación *final*), este script se enfoca en **crear ejemplos de frases que los usuarios podrían decir o escribir**, asociando cada frase a una **categoría de intención específica** (como "Saludo", "Insulto", "Respuesta Nombre", "Pregunta Económica", etc.). El objetivo es producir un dataset amplio y variado que permita al modelo de Procesamiento de Lenguaje Natural (PLN) aprender a identificar correctamente la intención detrás de las diversas formas en que un usuario puede expresarse. La **naturalidad y diversidad** de este dataset sintético son críticas para la robustez y precisión del clasificador de intenciones del chatbot en producción.
""")
st.info("Nota: Este script genera los datos de entrenamiento para el modelo que interpreta *lo que el usuario quiere decir* (`predict_intent` en `chatbot_logic.py`), no para el modelo que clasifica a la empresa.", icon="ℹ️")

st.header("Desglose Técnico del Código")

st.subheader("1. Importaciones y Configuración Inicial")

st.markdown("""
Se importan librerías esenciales: `random` para la aleatoriedad en la generación; `pandas` para la manipulación final del dataset; `Faker` para generar datos ficticios realistas (nombres, empresas); `num2words` para convertir números a formato textual, añadiendo una capa importante de naturalidad lingüística; `os` para manejo de rutas y archivos; y `numpy` para operaciones numéricas auxiliares. La sección de configuración define parámetros clave como el número de ejemplos a generar por categoría (`NUM_EJEMPLOS_POR_CATEGORIA`), el locale para `Faker` (`LOCALE = 'es'`), la ruta del archivo CSV de salida (`OUTPUT_CSV_FILE`), un diccionario (`FILES`) mapeando claves internas a rutas de archivos `.txt` que contienen plantillas o listas de palabras, y la lista explícita de `CATEGORIAS` de intención que se generarán.
""")
st.code("""
import random
import pandas as pd
from faker import Faker
from num2words import num2words
import os
import numpy as np

# --- Configuración ---
NUM_EJEMPLOS_POR_CATEGORIA = 1500
LOCALE = 'es'
OUTPUT_CSV_FILE = 'Data/dataset_completo.csv'
FILES = { # Mapeo de nombres internos a rutas de archivos TXT
    'insultos_sustantivos': 'Data/insultos.txt',
    # ... (otras rutas) ...
    'categorias_sectores': 'Data/Categorias-Empresa.txt'
}
CATEGORIAS = [ # Lista de intenciones a generar
    "Insulto", "Pregunta Económica", "Respuesta Nombre",
    # ... (lista completa de categorías) ...
]
fake = Faker(LOCALE)
""", language="python")


st.subheader("2. Funciones Auxiliares y Carga de Recursos")

st.markdown("""
Se definen funciones auxiliares para leer datos externos: `leer_categorias_sectores_txt` (similar a la vista en `generate_dataset.py`, adaptada para este script) y `leer_plantillas_txt` (una función genérica para leer archivos de texto línea por línea, usada para cargar saludos, insultos, etc.). Se incluye también la función `numero_a_palabras` que encapsula la funcionalidad de `num2words` para convertir números a texto en español, manejando números grandes y posibles errores. Posteriormente, se procede a cargar los datos: se lee el archivo de categorías/sectores y se itera sobre el diccionario `FILES` para cargar el contenido de los demás archivos `.txt` en un diccionario `plantillas_cargadas`. Además, se definen directamente en el código listas extensas (`plantillas_personales_internas_dinamicas`, `listas_fillers`, `intensificadores`, `verbos_negativos_inf`, `contextos_negativos`, etc.) y plantillas de respuesta específicas para cada categoría (`plantillas_respuesta_nombre`, `plantillas_respuesta_empleados`, etc.). Esta combinación de plantillas externas e internas, junto con las listas de relleno detalladas, forma la **base para generar diversidad lingüística**. Las listas de fillers (como `[tema_info]`, `[cualidad_humana]`, `[opcion_a]`, `[opcion_b]`) son particularmente importantes para crear variaciones semánticas dentro de estructuras de frases similares.
""")
st.code("""
def leer_categorias_sectores_txt(ruta_archivo_txt):
    # ... (código) ...
def leer_plantillas_txt(filename):
    # ... (código) ...
def numero_a_palabras(numero):
    # ... (código con num2words) ...

# --- Cargar Datos y Plantillas ---
df_categorias_global = leer_categorias_sectores_txt(FILES['categorias_sectores'])
plantillas_cargadas = {}
for key, filename in FILES.items():
    # ... (carga de archivos TXT) ...

# Definición directa de plantillas internas y listas de relleno muy extensas
plantillas_personales_internas_dinamicas = [...]
listas_fillers = {'[tema_info]': [...], '[tema_abstracto]': [...], ...}
intensificadores = [...]
verbos_negativos_inf = [...]
# ... (muchas más plantillas y listas) ...
plantillas_respuesta_nombre = [...]
plantillas_respuesta_empleados = [...]
# ... etc ...
""", language="python")


st.subheader("3. Generación de Frases (`generar_frase`)")

st.markdown("""
La función `generar_frase` es el núcleo de la generación sintética. Recibe una `categoria` de intención y el DataFrame de categorías/sectores como entrada. Su lógica se ramifica según la categoría solicitada, implementando diferentes estrategias para lograr naturalidad y variación:

* **Combinación de Elementos (Ej: "Insulto"):** Para categorías como "Insulto", selecciona aleatoriamente elementos de diferentes listas (sustantivos negativos, adjetivos negativos, intensificadores, verbos negativos, contextos negativos) y los inserta en diversas plantillas de frases predefinidas. Esto genera una amplia gama de expresiones ofensivas con estructuras variadas.
* **Plantillas con Placeholders (Ej: "Pregunta Económica", "Pregunta Personal", "Pregunta sobre Proceso"):** Utiliza plantillas (tanto internas como externas) que contienen placeholders (ej: `{campo_empresa_preg}`, `[tema_ai]`, `{sentimiento_usuario}`). Estos placeholders se rellenan dinámicamente seleccionando elementos al azar de las listas de `fillers` correspondientes o generando datos con `Faker`. Esto permite crear preguntas o afirmaciones que varían en su contenido específico pero mantienen la estructura de la intención.
* **Formato Numérico Variable (Ej: "Respuesta Empleados", "Respuesta Ganancias", etc.):** Para respuestas que involucran números, introduce aleatoriedad en el formato. Con una probabilidad definida (aquí, 50%), el número se presenta como dígitos formateados (con separadores de miles y símbolo de moneda) o se convierte a palabras usando `numero_a_palabras` y se combina con unidades textuales (ej: "mil quinientos pesos", "aproximadamente veinte millones de dólares"). Esta variación es crucial para que el modelo de PLN no dependa de un único formato numérico.
* **Uso de `Faker` (Ej: "Respuesta Nombre", "Saludo con_nombre"):** Integra `Faker` para generar nombres de empresas o personas realistas, insertándolos en plantillas de respuesta o saludo.
* **Decisiones Probabilísticas:** Emplea `random.random()` o `random.choices` para decidir entre diferentes sub-estrategias dentro de una categoría (ej: saludo simple vs. saludo con nombre; usar plantilla de archivo vs. plantilla interna dinámica), introduciendo otra capa de variabilidad.
* **Plantillas Múltiples por Categoría:** Para cada intención de respuesta (Nombre, Empleados, Deudas, etc.), se define una *amplia lista* de plantillas con diferentes estructuras sintácticas y tonos (formal, informal, entusiasta, reticente). El script selecciona una al azar, asegurando que la misma información (ej: número de empleados) se presente de múltiples maneras lingüísticas.

Esta combinación de técnicas busca maximizar la diversidad léxica y estructural de las frases generadas para cada intención, aproximándose a la variabilidad del lenguaje natural y mejorando la capacidad de generalización del modelo de clasificación de intenciones.
""")
st.code("""
def generar_frase(categoria, df_categorias):
    # ... (Lógica condicional por categoría) ...

    if categoria == "Insulto":
        # ... (Selecciona sust, adj, intens, verbo, contexto; elige plantilla; formatea) ...
        return insulto_generado

    elif categoria == "Pregunta Económica":
        # ... (Elige plantilla; selecciona fillers de listas; formatea) ...
        return pregunta_generada

    elif categoria == "Saludo":
        # ... (Decide tipo_saludo; usa plantilla TXT, con nombre Faker, o combinada) ...
        return saludo_generado

    elif categoria == "Pregunta Personal":
        # ... (Decide usar TXT o interna; elige plantilla; rellena placeholders de listas_fillers) ...
        return frase_final

    # ... (Lógica similar para Pregunta sobre Proceso) ...

    elif categoria == "Respuesta Nombre":
        # ... (Elige plantilla; usa fake.company()) ...
        return plantilla.format(nombre_empresa=fake.company())

    # ... (Lógica para Respuesta Categoria Empresa, Respuesta Sector) ...

    elif categoria == "Respuesta Empleados":
        # ... (Genera número; decide formato dígitos/texto; elige plantilla; formatea) ...
        return plantilla.format_map({'valor': valor_formateado})

    elif categoria in ["Respuesta Ganancias", ... , "Respuesta Deudas"]:
        # ... (Genera número; decide formato dígitos/texto con moneda/unidad; elige plantilla; formatea) ...
        # ... (Manejo especial para 0 deudas) ...
        return plantilla.format_map({'valor': valor_formateado})

    else:
        # ... (Fallback) ...
""", language="python")


st.subheader("4. Bucle Principal de Generación y Guardado")

st.markdown("""
El bloque final del script itera sobre la lista `CATEGORIAS` definida al inicio. Para cada categoría, entra en un bucle `while` que llama repetidamente a `generar_frase` hasta alcanzar el `NUM_EJEMPLOS_POR_CATEGORIA` deseado o un número máximo de intentos. Una característica importante aquí es el uso de un `set` (`frases_generadas_set`) para **garantizar la unicidad de las frases generadas**. Antes de añadir una frase al `dataset` final, se verifica si ya existe en el `set`. Si no existe, se añade tanto al `set` como a la lista `dataset`. Esto evita la redundancia excesiva en los datos de entrenamiento. Una vez generados todos los ejemplos, el `dataset` (una lista de diccionarios) se convierte en un DataFrame de `pandas`, se mezcla aleatoriamente (`df.sample(frac=1)`) para evitar cualquier sesgo por el orden de generación, y finalmente se guarda en el archivo CSV especificado (`OUTPUT_CSV_FILE`) usando codificación UTF-8. Se imprime información básica sobre el dataset resultante (tamaño, cabecera, distribución de categorías) para verificación.
""")
st.code("""
# --- Generación del Dataset ---
dataset = []
frases_generadas_set = set() # Para verificar unicidad
for categoria in CATEGORIAS:
    # ... (Bucle while para generar ejemplos por categoría) ...
        frase = generar_frase(categoria, df_categorias_global)
        if frase and isinstance(frase, str) and frase not in frases_generadas_set and ...: # Chequeo de validez y unicidad
            frases_generadas_set.add(frase) # Añadir al set
            dataset.append({'Frase': frase, 'Categoria': categoria}) # Añadir al resultado
            # ... (incrementar contador) ...
    # ... (Manejo de advertencias si no se generan suficientes ejemplos) ...

# --- Crear, Mezclar y Guardar DataFrame ---
if dataset:
    df = pd.DataFrame(dataset)
    df = df.sample(frac=1).reset_index(drop=True) # Mezclar
    df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8') # Guardar
    # ... (Imprimir información del dataset) ...
""", language="python")


st.header("Conclusión e Impacto en el Chatbot")

st.success("""
El script `create_responses_dataset.py` es un componente vital en la **creación de datos de entrenamiento para el modelo de clasificación de intenciones** del chatbot. Mediante el uso combinado de plantillas múltiples, la librería `Faker`, conversión de números a palabras, listas de relleno contextuales, decisiones probabilísticas y la garantía de unicidad, **intenta generar un corpus sintético que refleje la diversidad y naturalidad del lenguaje humano**. La calidad, volumen y, sobre todo, la **variedad lingüística** de los datos producidos en `dataset_completo.csv` determinarán en gran medida la capacidad del chatbot para comprender correctamente una amplia gama de entradas del usuario durante la interacción real. Un dataset bien generado aquí se traduce en una mejor función `predict_intent` en la lógica del chatbot.
""")
