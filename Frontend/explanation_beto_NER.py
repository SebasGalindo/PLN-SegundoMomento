import streamlit as st

st.set_page_config(page_title="Explicación: BETO_Clasificacion_Frases.ipynb", layout="wide")

st.title("Análisis del Script: `BETO_Extraccion_Caracteristicas.ipynb` (Entrenamiento del Modelo NER)")

st.header("Propósito General y Rol en el Chatbot")

st.markdown("""
Este Jupyter Notebook tiene como finalidad **entrenar un modelo de Reconocimiento de Entidades Nombradas (NER)** utilizando una técnica avanzada de Procesamiento de Lenguaje Natural (PLN) conocida como **fine-tuning (ajuste fino)** sobre un modelo pre-entrenado basado en la arquitectura **Transformer**, específicamente **BETO** (`dccuchile/bert-base-spanish-wwm-uncased`), una versión de BERT optimizada y pre-entrenada para el español.

**¿Por qué es crucial para el chatbot?** El chatbot necesita extraer información específica y estructurada de las respuestas en lenguaje natural del usuario. Por ejemplo, si el usuario dice "Mi empresa se llama Soluciones Andinas SAS y tenemos unos 50 empleados", el chatbot necesita identificar "Soluciones Andinas SAS" como `NOMBRE_EMPRESA` y "50" como `NUM_EMPLEADOS`. El modelo entrenado en este notebook es el responsable de realizar esta **extracción de entidades**. El resultado final de este notebook es un **modelo NER afinado y guardado** (`modelo_ner_frases`), que luego es cargado y utilizado por la función `predict_ner` en el script principal del chatbot (`chatbot_logic.py`) para procesar las respuestas del usuario durante la conversación.
""")

st.subheader("Entendiendo BERT/BETO y NER")
st.markdown("""
**BERT (Bidirectional Encoder Representations from Transformers)** es un modelo de lenguaje que revolucionó el PLN. Su arquitectura **Transformer** utiliza mecanismos de **auto-atención (self-attention)** para ponderar la importancia de diferentes palabras en una secuencia al representar una palabra específica. Es **bidireccional**, lo que significa que considera el contexto tanto a la izquierda como a la derecha de una palabra para entender su significado (a diferencia de modelos anteriores que leían de izquierda a derecha o viceversa). BERT se pre-entrena en enormes cantidades de texto (como Wikipedia y libros) en tareas como predecir palabras ocultas (Masked Language Model) y predecir si dos frases son consecutivas. Este pre-entrenamiento le da un conocimiento profundo de la gramática, la semántica y el contexto del lenguaje.

**BETO** es una variante de BERT pre-entrenada específicamente con un gran corpus en español, lo que lo hace ideal para tareas en este idioma.

Para **NER (Named Entity Recognition)**, se toma el modelo BETO pre-entrenado y añadimos una **capa de clasificación** adicional encima de las representaciones de salida de cada token. Durante el **fine-tuning**, se entrena *todo* el modelo (o principalmente la nueva capa y se ajusta ligeramente las capas inferiores) con un dataset específico de NER (en este caso, `dataset_ner.jsonl`). El objetivo es que el modelo aprenda a asignar la etiqueta NER correcta (como `B-NOMBRE_EMPRESA`, `I-VALOR_DEUDAS`, `O` para 'Outside') a cada **token** de la secuencia de entrada.
""")

st.header("Desglose Técnico del Código y Relación con BERT/PLN")

st.subheader("1. Instalación e Importaciones")
st.markdown("""
Se instalan y/o actualizan las librerías necesarias, principalmente `transformers` (de Hugging Face, que proporciona la implementación de BERT/BETO y las herramientas de entrenamiento), `torch` (el framework de deep learning sobre el que corre el modelo), `datasets` (para manejar los datos de forma eficiente) y `seqeval` (una librería estándar para evaluar tareas de etiquetado de secuencias como NER).
""")
st.code("""
import json
import os
import torch
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value, ClassLabel, Sequence
from transformers import (
    AutoTokenizer,                     # Para cargar el tokenizador específico de BETO
    AutoModelForTokenClassification, # Para cargar la arquitectura BERT con cabeza de clasificación de tokens
    TrainingArguments,                 # Para definir los hiperparámetros de entrenamiento
    Trainer,                           # Para manejar el bucle de entrenamiento y evaluación
    DataCollatorForTokenClassification # Para crear lotes de datos correctamente formateados
)
from seqeval.metrics import classification_report, f1_score # Métrica estándar para NER
from seqeval.scheme import IOB2 # Esquema de etiquetado BIO que usaremos
import traceback
""", language="python")

st.subheader("2. Configuración")
st.markdown("""
Se definen variables clave para el proceso:
* `ARCHIVO_DATASET_JSONL`: La ruta al archivo que contiene los datos de entrenamiento para NER. Este archivo debe tener frases y las entidades anotadas con sus posiciones (inicio, fin) y etiquetas.
* `MODELO_BERT_NER`: El identificador del modelo pre-entrenado a usar (BETO en este caso). Es fundamental elegir un modelo pre-entrenado en el idioma correcto (español).
* `OUTPUT_DIR_NER`: El directorio donde se guardará el modelo afinado, el tokenizador y los archivos de configuración. Este es el directorio que referenciará el chatbot.
* `MAX_LEN_NER`: Longitud máxima de la secuencia (en tokens) que BERT procesará. Textos más largos serán truncados. BERT tiene un límite inherente (usualmente 512 tokens), pero usar valores menores puede acelerar el entrenamiento si las frases no son extremadamente largas.
* `BATCH_SIZE_NER`: Número de ejemplos procesados simultáneamente durante el entrenamiento. Afecta el uso de memoria y la estabilidad del entrenamiento.
* `EPOCHS_NER`: Número de veces que el modelo verá el dataset completo durante el entrenamiento.
* `LEARNING_RATE_NER`: Tasa de aprendizaje para el optimizador. Controla cuán grandes son los ajustes a los pesos del modelo en cada paso. Es un hiperparámetro crucial.
* `LABEL_ALL_SUBWORDS`: Define cómo se etiquetan los subtokens (ver sección de tokenización).
""")
st.code("""
# --- Configuración ---
ARCHIVO_DATASET_JSONL = '/kaggle/input/pln-segundo-momento/Data/dataset_ner.jsonl' # <-- RUTA CRÍTICA
MODELO_BERT_NER = 'dccuchile/bert-base-spanish-wwm-uncased' # Modelo BETO
OUTPUT_DIR_NER = '/kaggle/working/modelo_ner_frases'       # Directorio de salida
MAX_LEN_NER = 128
BATCH_SIZE_NER = 8
EPOCHS_NER = 5
LEARNING_RATE_NER = 3e-5
LABEL_ALL_SUBWORDS = False # Importante para la alineación de etiquetas

os.makedirs(OUTPUT_DIR_NER, exist_ok=True)
# ... (prints de configuración) ...
""", language="python")

st.subheader("3. Carga y Preparación del Dataset (`cargar_datos_jsonl`, `crear_dataset_hf`)")
st.markdown("""
El primer paso es cargar los datos del archivo `.jsonl`. La función `cargar_datos_jsonl` lee este archivo, donde se espera que cada línea sea un JSON con al menos una clave `"text"` (la frase) y una clave `"entities"` (una lista de entidades, cada una representada como `[start_char, end_char, label_str]`).

Luego, `crear_dataset_hf` transforma esta lista de diccionarios Python en un objeto `Dataset` de la librería `datasets`. Esto es importante porque esta librería está optimizada para trabajar eficientemente con los modelos de `transformers`. Durante este proceso:
1.  Se extraen todos los textos y las listas de entidades.
2.  Se identifican todas las etiquetas únicas de entidades presentes en los datos (ej: 'NOMBRE_EMPRESA', 'VALOR_GANANCIAS').
3.  Se define la estructura (`Features`) del dataset, especificando que 'text' es un string, y 'ner_tags' es una *secuencia* de diccionarios, donde cada diccionario tiene 'start', 'end' y 'label'. Crucialmente, el 'label' se define como `ClassLabel(names=label_list)`, lo que mapea automáticamente las etiquetas de texto a IDs numéricos internos, algo necesario para el modelo.
4.  Se crea el `Dataset` usando `Dataset.from_dict`.

**Relevancia para BERT:** BERT necesita datos estructurados para el fine-tuning. Aunque el input inicial son textos y spans de caracteres, esta preparación los organiza en un formato compatible con las herramientas de Hugging Face y define las posibles clases de salida (las etiquetas de entidad).
""")
st.code("""
def cargar_datos_jsonl(ruta_archivo):
    # ... (Lectura de JSONL, manejo de errores) ...
    return datos # Lista de diccionarios [{'text': '...', 'entities': [[s,e,l], ...]}, ...]

def crear_dataset_hf(datos_cargados):
    # ... (Extrae textos y listas de entidades) ...
    # ... (Validaciones de formato de entidades) ...
    label_list = sorted(list(all_labels_set)) # Lista única de etiquetas (ej: ['NOMBRE_EMPRESA', ...])

    # Define la estructura esperada por Hugging Face
    features = Features({
        'id': Value('string'),
        'text': Value('string'),
        'ner_tags': Sequence({ # Indica una lista de entidades por ejemplo
            'start': Value('int32'),
            'end': Value('int32'),
            'label': ClassLabel(names=label_list) # Mapea texto a ID automáticamente
        })
    })

    hf_dataset = Dataset.from_dict({ # Crea el objeto Dataset
        "id": [str(i) for i in range(len(textos))],
        "text": textos,
        "ner_tags": ner_tags_list, # Lista de listas de dicts {'start':s, 'end':e, 'label':ID}
    }, features=features)

    return hf_dataset, label_list
# --- Cargar y Crear ---
datos_originales = cargar_datos_jsonl(ARCHIVO_DATASET_JSONL)
if datos_originales:
    dataset_hf, label_list = crear_dataset_hf(datos_originales)
# ... (Manejo de errores) ...
""", language="python")


st.subheader("4. Creación de Etiquetas BIO y Mapeos")
st.markdown("""
BERT, para tareas de NER (Token Classification), no predice simplemente la etiqueta de la entidad (como 'NOMBRE_EMPRESA'). Predice una etiqueta para *cada token*. Para manejar entidades que abarcan múltiples tokens (ej: "Soluciones Andinas SAS"), se utiliza un esquema de etiquetado como **BIO (Beginning, Inside, Outside)**.
* `B-LABEL`: Marca el **inicio** de una entidad de tipo LABEL.
* `I-LABEL`: Marca un token que está **dentro** de una entidad de tipo LABEL, pero no es el primero.
* `O`: Marca un token que está **fuera** de cualquier entidad nombrada.

Este código toma la `label_list` obtenida del paso anterior (ej: `['NOMBRE_EMPRESA', 'VALOR_DEUDAS']`) y crea la lista completa de etiquetas BIO (`bio_label_list`): `['O', 'B-NOMBRE_EMPRESA', 'I-NOMBRE_EMPRESA', 'B-VALOR_DEUDAS', 'I-VALOR_DEUDAS']`. Luego, crea dos diccionarios de mapeo:
* `label2id`: Convierte una etiqueta BIO string (ej: "B-NOMBRE_EMPRESA") a un ID numérico único.
* `id2label`: Hace la conversión inversa (ID numérico a string).

**Relevancia para BERT:** El modelo BERT aprenderá a predecir estos IDs numéricos correspondientes a las etiquetas BIO para cada token de entrada. Estos mapeos son esenciales para codificar las etiquetas durante el entrenamiento y decodificar las predicciones del modelo durante la evaluación y la inferencia. Se guardan en `ner_label_mappings.json` para ser usados por el chatbot.
""")
st.code("""
if label_list:
    # Crear etiquetas BIO a partir de las etiquetas base
    bio_label_list = ["O"] + [f"B-{lbl}" for lbl in label_list] + [f"I-{lbl}" for lbl in label_list]
    # Crear mapeos
    label2id = {label: i for i, label in enumerate(bio_label_list)}
    id2label = {i: label for i, label in enumerate(bio_label_list)}
    num_labels_ner = len(bio_label_list)

    # ... (Guardar mapeos en ner_label_mappings.json) ...
else:
    # ... (Manejo de error) ...
""", language="python")

st.subheader("5. Tokenización y Alineación de Etiquetas (`AutoTokenizer`, `tokenize_and_align_labels`)")
st.markdown("""
Este es uno de los pasos más técnicos y cruciales del preprocesamiento para BERT en tareas de NER.

**a) Tokenización:**
* `AutoTokenizer.from_pretrained(MODELO_BERT_NER)`: Carga el **tokenizador específico** asociado al modelo BETO. BERT no opera sobre palabras, sino sobre **subtokens** (WordPieces). Por ejemplo, "Banco Agrario" podría tokenizarse como `['[CLS]', 'ban', '##co', 'agrar', '##io', '[SEP]']`. `[CLS]` y `[SEP]` son tokens especiales añadidos por BERT.
* `tokenizer(...)`: Aplica el tokenizador al lote de textos.
    * `truncation=True, padding="max_length", max_length=MAX_LEN_NER`: Asegura que todas las secuencias tokenizadas tengan exactamente `MAX_LEN_NER` tokens, truncando las más largas y añadiendo tokens de padding (`[PAD]`) a las más cortas. BERT requiere entradas de longitud fija.
    * `return_offsets_mapping=True`: ¡Fundamental! Devuelve el mapeo de cada token a sus caracteres de inicio y fin en el texto *original*. Esto es lo que permite alinear las etiquetas.
* **Relación con PLN:** La **tokenización** (específicamente subword tokenization) es un paso indispensable. BERT opera a nivel de estos (sub)tokens, no de palabras completas.

**b) Alineación de Etiquetas (`tokenize_and_align_labels`):**
El desafío es que las etiquetas originales están a nivel de *caracteres* (inicio, fin) y el modelo predice a nivel de *tokens*. Esta función alinea las etiquetas BIO con los tokens/subtokens generados:
1.  **Iteración:** Procesa cada ejemplo (frase) en el lote.
2.  **Reconstrucción de Entidades:** Accede a los datos `ner_tags` del ejemplo actual (que ahora contienen IDs numéricos de `ClassLabel` gracias a `crear_dataset_hf`).
3.  **Mapeo Palabra-Token:** Obtiene `word_ids` (que mapea cada token a su palabra original) y `offset_mapping` (token a caracteres originales).
4.  **Inicialización:** Crea una lista de etiquetas `label_ids_for_example` inicializada con `-100`. Este valor especial le indica a PyTorch/BERT que ignore estos tokens al calcular la función de pérdida (loss) durante el entrenamiento. Se usa para tokens especiales (`[CLS]`, `[SEP]`, `[PAD]`) y, opcionalmente, para subtokens.
5.  **Lógica de Alineación BIO:** Itera sobre cada `token_idx`.
    * Si es un token especial o de padding, asigna `-100`.
    * Si es un **subtoken** (pertenece a la misma palabra que el token anterior):
        * Si `LABEL_ALL_SUBWORDS` es `False` (lo común y usado aquí), se le asigna `-100` (solo se etiqueta el primer subtoken de una palabra).
        * Si fuera `True`, se le asignaría la etiqueta `I-LABEL` correspondiente a la etiqueta del primer subtoken (que sería `B-LABEL` o `I-LABEL`).
    * Si es el **inicio de una nueva palabra** (no subtoken):
        * Usa el `offset_mapping` del token para ver si cae dentro de alguna de las entidades anotadas (`sorted_ner_tags`).
        * Si **no** cae en ninguna entidad, se le asigna la etiqueta `O` (convertida a su ID).
        * Si **sí** cae en una entidad (ej: 'NOMBRE_EMPRESA'):
            * Verifica si el token útil *anterior* también pertenecía a la **misma** entidad.
            * Si era la misma entidad, se asigna la etiqueta `I-` (ej: `I-NOMBRE_EMPRESA`, convertida a ID).
            * Si no era la misma o era `O`, se asigna la etiqueta `B-` (ej: `B-NOMBRE_EMPRESA`, convertida a ID), marcando el inicio de la entidad.
6.  **Resultado:** La función devuelve un diccionario `tokenized_inputs` que ahora contiene la clave `"labels"` con la lista de IDs de etiquetas BIO alineadas para cada token de cada ejemplo del lote.

**Relevancia para BERT:** BERT *necesita* que las etiquetas estén alineadas perfectamente con los tokens que procesa. Esta función realiza esa traducción indispensable entre las anotaciones a nivel de caracteres y las predicciones a nivel de token, implementando el esquema BIO y manejando la complejidad de la tokenización por subpalabras.
""")
st.code("""
# --- Cargar Tokenizador ---
tokenizer_ner = AutoTokenizer.from_pretrained(MODELO_BERT_NER, use_fast=True)

# --- Función de Tokenización y Alineación ---
def tokenize_and_align_labels(examples):
    # Tokenizar lote de textos
    tokenized_inputs = tokenizer_ner(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN_NER,
        return_offsets_mapping=True, # CLAVE para alinear
        is_split_into_words=False # El tokenizer divide las palabras
    )

    all_aligned_labels = []
    num_examples_in_batch = len(examples["text"])

    for i in range(num_examples_in_batch): # Iterar sobre cada ejemplo
        # ... (Reconstrucción de ner_tags para el ejemplo i desde examples['ner_tags'][i]) ...
        # ... (Obtener word_ids y offset_mapping para el ejemplo i) ...

        label_ids_for_example = [-100] * len(word_ids) # Inicializar con ignore_index

        # ... (Lógica detallada de iteración sobre tokens) ...
        #     Si es token especial/padding -> -100
        #     Si es subword y LABEL_ALL_SUBWORDS==False -> -100
        #     Si es inicio de palabra:
        #         Buscar entidad correspondiente usando offset_mapping y ner_tags_reconstructed
        #         Si NO está en entidad -> label_ids_for_example[token_idx] = label2id["O"]
        #         Si ESTÁ en entidad:
        #             Verificar etiqueta del token útil anterior
        #             Si continúa misma entidad -> label_ids_for_example[token_idx] = label2id[f"I-{base_label_str}"]
        #             Si empieza entidad o es diferente -> label_ids_for_example[token_idx] = label2id[f"B-{base_label_str}"]
        # ... (Fin de lógica de alineación) ...

        all_aligned_labels.append(label_ids_for_example)

    tokenized_inputs["labels"] = all_aligned_labels # Añadir etiquetas alineadas
    return tokenized_inputs

# Aplicar la función a todo el dataset
if dataset_hf:
    tokenized_dataset = dataset_hf.map(
        tokenize_and_align_labels,
        batched=True, # Procesar en lotes es más eficiente
        remove_columns=dataset_hf.column_names # Eliminar columnas originales no necesarias
    )
    # ... (Manejo de errores y prints) ...
""", language="python")

st.subheader("6. División del Dataset y Carga del Modelo")
st.markdown("""
El `tokenized_dataset` se divide en conjuntos de entrenamiento y validación (`train_test_split`) usando `DatasetDict`. Esto es estándar en Machine Learning para evaluar el rendimiento del modelo en datos que no ha visto durante el entrenamiento y detectar overfitting.

Luego, se carga el modelo BERT/BETO pre-entrenado usando `AutoModelForTokenClassification.from_pretrained`. Es crucial:
1.  Pasar el `MODELO_BERT_NER` correcto.
2.  Indicar `num_labels=num_labels_ner`: El modelo necesita saber cuántas posibles etiquetas BIO de salida hay para crear la capa de clasificación final con el tamaño adecuado.
3.  Pasar los mapeos `id2label` y `label2id`: Ayuda al modelo a inicializarse correctamente y es útil para la inferencia posterior.
4.  Mover el modelo al dispositivo correcto (`.to(device_ner)`) para aprovechar la GPU si está disponible.

**Relevancia para BERT:** Se carga la arquitectura BERT pre-entrenada y se le añade la "cabeza" específica para clasificación de tokens, lista para ser afinada. La división de datos es necesaria para un entrenamiento robusto.
""")
st.code("""
# Dividir en entrenamiento y validación
train_val_split = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
final_datasets = DatasetDict({
    'train': train_val_split['train'],
    'validation': train_val_split['test']
})

# --- Cargar Modelo NER ---
model_ner = AutoModelForTokenClassification.from_pretrained(
    MODELO_BERT_NER,
    num_labels=num_labels_ner, # Número correcto de etiquetas BIO
    id2label=id2label,       # Mapeo ID -> Label BIO
    label2id=label2id        # Mapeo Label BIO -> ID
)
# --- Configurar Dispositivo ---
device_ner = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_ner.to(device_ner)
""", language="python")

st.subheader("7. Configuración del Entrenamiento (`TrainingArguments`, `DataCollator`, `compute_metrics`)")
st.markdown("""
Se definen los parámetros y herramientas para el proceso de fine-tuning:
* `TrainingArguments`: Configura todos los hiperparámetros y estrategias: directorio de salida, número de épocas, tamaño de lote, tasa de aprendizaje, decaimiento de peso (weight decay, una técnica de regularización), cada cuánto evaluar (`eval_strategy`), cada cuánto guardar (`save_strategy`), si cargar el mejor modelo al final (`load_best_model_at_end`), qué métrica usar para decidir cuál es el mejor (`metric_for_best_model`, aquí F1-score ponderado), etc.
* `DataCollatorForTokenClassification`: Es un objeto auxiliar que toma una lista de ejemplos del dataset (ya tokenizados y alineados) y los agrupa inteligentemente en un lote (batch) listo para ser alimentado al modelo. Se encarga principalmente de aplicar padding dinámico a las secuencias `input_ids`, `attention_mask` y `labels` dentro de cada lote para que todas tengan la misma longitud (la del ejemplo más largo *en ese lote*), lo cual es más eficiente que hacer padding a `MAX_LEN_NER` en el paso de tokenización.
* `compute_metrics`: Define la función que se usará durante la evaluación. Recibe las predicciones del modelo (logits) y las etiquetas verdaderas. Realiza los siguientes pasos:
    1.  Convierte los logits a IDs predichos (`np.argmax`).
    2.  Decodifica tanto los IDs predichos como los IDs verdaderos a etiquetas BIO string (usando `id2label`), **ignorando las etiquetas `-100`** (correspondientes a tokens especiales o subtokens ignorados).
    3.  Utiliza `seqeval.metrics.classification_report` con el esquema `IOB2` para calcular métricas a nivel de entidad (Precision, Recall, F1-score) de forma robusta. Devuelve un diccionario con las métricas clave (ej: `f1_weighted`).

**Relevancia para BERT:** `TrainingArguments` controla el proceso de aprendizaje. El `DataCollator` asegura que los datos lleguen al modelo en el formato correcto por lotes. `compute_metrics` define cómo se mide el rendimiento del modelo específicamente para la tarea NER usando métricas estándar del campo (seqeval).
""")
st.code("""
# --- Configuración del Entrenamiento ---
training_args_ner = TrainingArguments(
    output_dir=OUTPUT_DIR_NER,
    num_train_epochs=EPOCHS_NER,
    per_device_train_batch_size=BATCH_SIZE_NER,
    # ... (otros argumentos como learning_rate, eval_strategy, etc.) ...
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1_weighted", # Métrica para seleccionar el mejor checkpoint
    # ...
)

# --- Data Collator ---
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer_ner)

# --- Métrica de Evaluación (usando seqeval) ---
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Convertir IDs a etiquetas BIO strings, ignorando -100
    true_labels = [[id2label.get(l, 'O') for l in label if l != -100] for label in labels]
    true_predictions = [[id2label.get(p, 'O') for (p, l) in zip(prediction, label) if l != -100]
                       for prediction, label in zip(predictions, labels)]

    # Calcular métricas con seqeval
    report = classification_report(true_labels, true_predictions, output_dict=True, mode='strict', scheme=IOB2, zero_division=0)
    results = { # Extraer métricas relevantes
        "precision_weighted": report["weighted avg"]["precision"],
        "recall_weighted": report["weighted avg"]["recall"],
        "f1_weighted": report["weighted avg"]["f1-score"],
    }
    return results
""", language="python")


st.subheader("8. Inicialización y Ejecución del `Trainer`")
st.markdown("""
Se instancia la clase `Trainer`, que es la clase principal de Hugging Face para manejar el entrenamiento y la evaluación. Se le pasa el `model_ner` (el modelo a entrenar), los `training_args_ner`, los datasets de entrenamiento y validación (`final_datasets`), el `tokenizer_ner`, el `data_collator` y la función `compute_metrics`.

Finalmente, `trainer.train()` lanza el bucle de fine-tuning. El `Trainer` se encarga automáticamente de:
* Iterar sobre las épocas y los lotes de datos.
* Mover los datos al dispositivo correcto (GPU/CPU).
* Realizar el forward pass (pasar los datos por el modelo para obtener predicciones).
* Calcular la loss (función de pérdida) comparando predicciones y etiquetas verdaderas.
* Realizar el backward pass (calcular gradientes).
* Actualizar los pesos del modelo usando el optimizador (definido implícitamente, usualmente AdamW).
* Evaluar periódicamente en el set de validación usando `compute_metrics`.
* Guardar checkpoints del modelo según `save_strategy`.
* Cargar el mejor modelo encontrado al final si `load_best_model_at_end=True`.

**Relevancia para BERT:** El `Trainer` abstrae la complejidad del bucle de entrenamiento de PyTorch, permitiendo realizar el fine-tuning de BERT de manera eficiente y organizada, integrando todos los componentes preparados anteriormente.
""")
st.code("""
# --- Inicializar Trainer ---
trainer = Trainer(
    model=model_ner,
    args=training_args_ner,
    train_dataset=final_datasets["train"],
    eval_dataset=final_datasets["validation"],
    tokenizer=tokenizer_ner,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# --- Iniciar Entrenamiento ---
print("\\n--- Iniciando Entrenamiento NER ---")
train_result = trainer.train()
# ... (Guardar métricas finales) ...
""", language="python")


st.subheader("9. Guardado del Modelo Final y Tokenizador")
st.markdown("""
Una vez completado el entrenamiento, y dado que se usó `load_best_model_at_end=True`, el objeto `trainer` ahora contiene el modelo con los pesos del checkpoint que obtuvo el mejor rendimiento en el set de validación (según `metric_for_best_model`). Es crucial guardar este modelo final junto con su tokenizador y los mapeos de etiquetas para poder usarlo después en el chatbot.
* `trainer.save_model(OUTPUT_DIR_NER)`: Guarda los pesos del modelo afinado y su archivo de configuración.
* `tokenizer_ner.save_pretrained(OUTPUT_DIR_NER)`: Guarda los archivos del tokenizador (vocabulario, configuración, etc.).
* El archivo `ner_label_mappings.json` (con `id2label` y `label2id`) ya se guardó anteriormente.

**Relevancia para BERT:** Este paso crea los artefactos persistentes (archivos en disco) que encapsulan el modelo BERT afinado y todo lo necesario para usarlo posteriormente para inferencia (predicción) en el chatbot.
""")
st.code("""
# --- Guardar el Mejor Modelo y Tokenizador ---
print(f"\\nGuardando el modelo final (mejor modelo cargado) en: {OUTPUT_DIR_NER}")
trainer.save_model(OUTPUT_DIR_NER)
tokenizer_ner.save_pretrained(OUTPUT_DIR_NER)
# El mapeo de etiquetas ('ner_label_mappings.json') ya se guardó
print("¡Modelo NER y tokenizador guardados exitosamente!")
""", language="python")


st.subheader("10. Carga y Predicción (Inferencia)")
st.markdown("""
Esta sección demuestra cómo cargar el modelo recién guardado y usarlo para predecir entidades en texto nuevo.
* Se cargan los mapeos (`id2label_cargado`, `label2id_cargado`) desde el archivo JSON.
* Se cargan el modelo y el tokenizador desde el directorio `OUTPUT_DIR_NER` usando `AutoModelForTokenClassification.from_pretrained` y `AutoTokenizer.from_pretrained`. Hugging Face reconoce automáticamente los archivos guardados.
* Se mueve el modelo al dispositivo adecuado y se pone en modo evaluación (`model_cargado_ner.eval()`), lo cual desactiva capas como Dropout que solo se usan durante el entrenamiento.
* La función `predecir_ner` encapsula el proceso de inferencia:
    1.  **Tokeniza** el texto de entrada usando el tokenizador cargado, obteniendo `input_ids`, `attention_mask` y, crucialmente, `offset_mapping`.
    2.  Realiza la **predicción** pasando los tensores al modelo (`outputs = modelo(**inputs)`).
    3.  Obtiene los **IDs de etiqueta BIO predichos** para cada token (`torch.argmax`).
    4.  **Reconstruye las entidades:** Este es el proceso inverso a la alineación. Itera sobre los tokens predichos y sus `offset_mapping`. Usando las etiquetas BIO predichas (`B-`, `I-`, `O`), agrupa tokens consecutivos que pertenecen a la misma entidad y utiliza los `offset_mapping` del primer y último token de la entidad para extraer el *texto original* correspondiente a esa entidad desde la frase de entrada. Devuelve una lista de diccionarios, cada uno representando una entidad encontrada con su texto, etiqueta y posición.

**Relevancia para BERT:** Muestra el uso práctico del modelo BERT afinado. La tokenización y el uso de `offset_mapping` son de nuevo esenciales para mapear las predicciones a nivel de token de BERT de vuelta a spans de texto significativos en la entrada original. **Embeddings** son generados internamente por el modelo BERT cargado para cada token de entrada antes de pasar por la capa de clasificación.

**Relación con POS Tagging / Lematización:** Como se mencionó, BERT aprende implícitamente información gramatical y de raíces de palabras, pero este script **no** utiliza explícitamente POS tagging ni lematización como pasos de preprocesamiento o postprocesamiento para la tarea NER. Su enfoque se basa en la capacidad del Transformer para capturar contexto directamente desde los tokens/subtokens.
""")
st.code("""
# --- Cargar Modelo y Tokenizador Guardados para Inferencia ---
model_load_path = OUTPUT_DIR_NER
# ... (cargar mapeos desde json) ...
model_cargado_ner = AutoModelForTokenClassification.from_pretrained(model_load_path)
tokenizer_cargado_ner = AutoTokenizer.from_pretrained(model_load_path)
# ... (configurar dispositivo y model.eval()) ...

# --- Función de Predicción NER ---
def predecir_ner(texto, modelo, tokenizer, device, id_to_label_map, max_len):
    # 1. Tokenizar y obtener offsets
    inputs = tokenizer(texto, ..., return_offsets_mapping=True)
    offset_mapping = inputs.pop("offset_mapping").cpu().squeeze().tolist()
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 2. Inferencia
    with torch.no_grad(): # Importante para desactivar cálculo de gradientes
        outputs = modelo(**inputs)

    # 3. Obtener predicciones (IDs BIO)
    predictions = torch.argmax(outputs.logits, dim=-1).cpu().squeeze().tolist()

    # 4. Reconstruir entidades usando predicciones BIO y offset_mapping
    entities = []
    # ... (Lógica para iterar tokens, identificar B- y I-, usar offsets para extraer texto) ...
    return entities, texto

# --- Bucle de Chat Simple NER ---
# ... (Usa la función predecir_ner para probar el modelo cargado) ...

""", language="python")


st.header("Conclusión Final")

st.success("""
Este notebook implementa un flujo de trabajo completo y estándar para **afinar (fine-tuning) un modelo Transformer pre-entrenado (BETO) para la tarea de Reconocimiento de Entidades Nombradas (NER) en español**. Cubre desde la carga y preprocesamiento de datos (incluyendo la crucial tokenización y alineación de etiquetas BIO), pasando por la configuración y ejecución del entrenamiento usando las herramientas de Hugging Face (`Trainer`), hasta el guardado del modelo final y una demostración de cómo usarlo para extraer entidades de texto nuevo. El modelo resultante, guardado en `modelo_ner_frases`, es una pieza clave de inteligencia artificial que permite al chatbot financiero comprender y extraer datos estructurados de las respuestas del usuario. Su rendimiento depende directamente de la calidad del dataset de entrenamiento (`dataset_ner.jsonl`) y de la correcta ejecución de los pasos técnicos descritos, especialmente la alineación de etiquetas con la tokenización por subpalabras inherente a BERT.
""")

# Fin de la simulación de página Streamlit