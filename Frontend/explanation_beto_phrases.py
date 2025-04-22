import streamlit as st

st.set_page_config(page_title="Explicación: Notebook BETO (Clasificación de Intención)", layout="wide")

st.title("Análisis del Script: `BET_Clasificacion_Frases.ipynb` (Modelo de Clasificación de Intención)")
st.info("Nota: Esta explicación se centra en la **primera parte** del notebook, dedicada al entrenamiento del modelo de Clasificación de Intención.", icon="ℹ️")

st.header("Propósito General y Rol en el Chatbot")

st.markdown("""
Este Jupyter Notebook, en su primera sección, implementa el proceso de **fine-tuning (ajuste fino)** de un modelo Transformer pre-entrenado, específicamente **BETO** (`dccuchile/bert-base-spanish-wwm-uncased`), para una tarea de **Clasificación de Secuencias (Sequence Classification)**. El objetivo es entrenar un modelo capaz de **asignar una categoría de intención predefinida** (como "Saludo", "Insulto", "Respuesta Nombre", "Pregunta Económica", etc.) a una **frase completa** proporcionada por el usuario.

**Rol en el Chatbot:** Este modelo es el **primer intérprete** de lo que el usuario dice. Permite al chatbot entender el **propósito general** de la entrada del usuario. Por ejemplo, si el usuario escribe "hola", el modelo debe identificar la intención "Saludo"; si escribe "¿cuánto debo?", debe identificar "Pregunta Económica" (o una categoría similar); si escribe "mi empresa tiene 50 empleados", debe identificar "Respuesta Empleados". Esta clasificación inicial es **fundamental** porque dirige la lógica de diálogo subsiguiente: el chatbot sabe si debe saludar, responder a una pregunta, procesar un dato, manejar un insulto, etc. El artefacto resultante de esta sección es el modelo guardado en el directorio `modelo_clasificador_frases`, que es cargado y utilizado por la función `predict_intent` en el script principal del chatbot (`chatbot_logic.py`).
""")

st.subheader("Fundamentos: BERT/BETO para Clasificación de Secuencias")
st.markdown("""
La arquitectura **BERT (Bidirectional Encoder Representations from Transformers)**, y su versión en español **BETO**, procesa secuencias de texto completas de manera bidireccional. Emplea mecanismos de **auto-atención** para capturar las relaciones contextuales entre palabras (subtokens). El pre-entrenamiento masivo le otorga una comprensión robusta del lenguaje.

Para la tarea específica de **Clasificación de Secuencias**, se adapta la arquitectura de la siguiente forma:
1.  **Tokenización:** La frase completa del usuario se convierte en una secuencia de subtokens (WordPieces) mediante un tokenizador específico (el de BETO), añadiendo tokens especiales como `[CLS]` al inicio y `[SEP]` al final.
2.  **Procesamiento BERT:** La secuencia de subtokens se introduce en el modelo BERT, que genera una **representación vectorial contextualizada** para cada subtoken.
3.  **Representación Agregada ([CLS]):** Para clasificar *toda* la secuencia, se utiliza principalmente el vector de salida correspondiente al token especial `[CLS]`. Se considera que este vector, influenciado por todos los demás tokens de la secuencia a través de la auto-atención, encapsula el significado semántico global de la frase.
4.  **Cabeza de Clasificación:** Se añade una capa neuronal simple (generalmente una capa lineal seguida de una función de activación como Softmax) encima de la representación vectorial del `[CLS]`. El número de neuronas en la salida de esta capa lineal es igual al número de categorías de intención distintas que se quieren predecir.
5.  **Fine-tuning:** Durante el fine-tuning, se entrena el modelo (principalmente la nueva cabeza de clasificación, aunque los pesos de BERT también pueden ajustarse ligeramente) utilizando un dataset etiquetado (`dataset_completo.csv`), donde cada frase está asociada a su categoría de intención correcta. El modelo aprende a mapear la representación semántica capturada en el vector `[CLS]` a la categoría de intención correspondiente.
""")

st.header("Desglose Técnico del Código (Parte 1: Clasificación de Intenciones)")

st.subheader("1. Instalación e Importaciones")
st.markdown("""
Se instalan las librerías base como `transformers`, `torch`, `pandas`, `scikit-learn`, y `numpy`. Las importaciones clave para esta tarea son:
* `BertTokenizerFast`: Para cargar el tokenizador WordPiece asociado a BETO. Es crucial usar el tokenizador correspondiente al modelo pre-entrenado.
* `BertForSequenceClassification`: Esta clase de la librería `transformers` carga la arquitectura BERT pre-entrenada **junto con una cabeza (capa) de clasificación predefinida para tareas a nivel de secuencia**, que opera sobre la salida del token `[CLS]`.
* `train_test_split`: De `sklearn`, para dividir el dataset.
* `LabelEncoder`: De `sklearn`, para convertir las etiquetas de intención textuales a IDs numéricos que el modelo pueda procesar.
* `Dataset`, `DataLoader`: Clases de PyTorch para manejar los datos eficientemente durante el entrenamiento.
* `AdamW`, `get_linear_schedule_with_warmup`: Optimizador y planificador de tasa de aprendizaje, comunes para entrenar Transformers.
* `accuracy_score`, `f1_score`: Métricas estándar para evaluar el rendimiento de la clasificación.
""")
st.code("""
# Instalación (generalmente en la primera celda)
# %pip install transformers torch pandas scikit-learn numpy

# Importaciones
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizerFast, BertForSequenceClassification, get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import numpy as np
import json
import os
from tqdm.auto import tqdm # Para barras de progreso
from sklearn.metrics import accuracy_score, f1_score
""", language="python")

st.subheader("2. Configuración")
st.markdown("""
Se definen las rutas y los hiperparámetros del entrenamiento:
* `ARCHIVO_DATASET`: Apunta al archivo `dataset_completo.csv`, que contiene las frases de ejemplo y su `Categoria` (intención). Este dataset es el resultado del script `create_responses_dataset.py`.
* `MODELO_BERT`: Especifica el modelo pre-entrenado base (`dccuchile/bert-base-spanish-wwm-uncased` - BETO).
* `OUTPUT_DIR`: Directorio donde se guardarán los artefactos del modelo afinado (pesos, configuración, tokenizador, mapeo de etiquetas). Este directorio (`modelo_clasificador_frases`) será utilizado por el chatbot.
* `MAX_LEN`: Define la longitud máxima de las secuencias tokenizadas. Las frases más largas se truncarán y las más cortas se rellenarán (padding). Es un parámetro importante para BERT.
* `BATCH_SIZE`, `EPOCHS`, `LEARNING_RATE`: Hiperparámetros clave que controlan cómo se realiza el entrenamiento (cuántos ejemplos por lote, cuántas veces ver el dataset completo, y cuán rápido se ajustan los pesos).
""")
st.code("""
# --- Configuración ---
ARCHIVO_DATASET = 'drive/MyDrive/PLN_Segundo_Momento/dataset_completo.csv' # Archivo de frases/categorías
MODELO_BERT = 'dccuchile/bert-base-spanish-wwm-uncased' # Modelo BETO
OUTPUT_DIR = 'drive/MyDrive/PLN_Segundo_Momento/modelo_clasificador_frases'
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5

# Crear directorio de salida si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)
""", language="python")

st.subheader("3. Carga y Preprocesamiento del Dataset")
st.markdown("""
Se carga el dataset `dataset_completo.csv` usando pandas. Se eliminan filas con datos faltantes. El paso más significativo aquí es la **codificación de etiquetas**:
* `LabelEncoder`: Transforma las etiquetas de categoría textuales (ej: "Saludo") en representaciones numéricas enteras (ej: 0, 1, 2...). Esto es necesario porque los modelos neuronales operan sobre números.
* **Mapeo de Etiquetas:** Es **vital** guardar la correspondencia entre las etiquetas de texto y los IDs numéricos asignados por el `LabelEncoder`. Se crean los diccionarios `label2id` (Texto -> ID) y `id2label` (ID -> Texto) y se almacenan en un archivo JSON (`label_mappings.json`) en el directorio de salida. El chatbot usará este archivo para interpretar las predicciones numéricas del modelo y convertirlas de nuevo en la categoría de intención correspondiente.
* **División Entrenamiento/Validación:** El dataset se divide en conjuntos de entrenamiento y validación usando `train_test_split`. Se utiliza `stratify` para asegurar que la distribución de las diferentes categorías de intención sea similar en ambos conjuntos, lo cual es importante para una evaluación fiable, especialmente si algunas intenciones son menos frecuentes que otras.
""")
st.code("""
# --- Carga y Preprocesamiento ---
df = pd.read_csv(ARCHIVO_DATASET)
# ... (Limpieza de nulos/vacíos) ...

# --- Codificar las Etiquetas ---
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['Categoria'])
num_labels = len(label_encoder.classes_) # Número total de intenciones únicas

# Guardar el mapeo de label a id y viceversa
label2id = {label: i for i, label in enumerate(label_encoder.classes_)}
id2label = {i: label for i, label in enumerate(label_encoder.classes_)}
map_file_path = os.path.join(OUTPUT_DIR, 'label_mappings.json')
with open(map_file_path, 'w', encoding='utf-8') as f:
    json.dump({'label2id': label2id, 'id2label': id2label}, f, ensure_ascii=False, indent=4)
print(f"Mapeos de etiquetas guardados en: {map_file_path}")

# --- Dividir en Entrenamiento y Validación ---
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['Frase'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label'] # Asegura proporción de clases
)
""", language="python")

st.subheader("4. Tokenización Específica para BERT")
st.markdown("""
Este paso adapta el texto al formato requerido por BERT.
* `BertTokenizerFast.from_pretrained(MODELO_BERT)`: Carga el tokenizador **WordPiece** asociado con el modelo BETO. Es fundamental usar el tokenizador correcto para que las frases se dividan en los mismos subtokens que el modelo vio durante su pre-entrenamiento.
* **Proceso de Tokenización (PLN Core):** La función `tokenizer()` aplica varias operaciones clave:
    1.  **Tokenización por Subpalabras:** Divide las palabras en unidades más pequeñas (ej: "clasificación" -> `['clasi', '##ficacion']`). Esto permite manejar un vocabulario fijo y palabras desconocidas.
    2.  **Adición de Tokens Especiales:** Inserta `[CLS]` al principio y `[SEP]` al final de cada secuencia tokenizada. El token `[CLS]` es especialmente relevante para la clasificación de secuencias.
    3.  **Conversión a IDs:** Mapea cada subtoken y token especial a su ID numérico correspondiente en el vocabulario del modelo.
    4.  **Padding y Truncamiento:** Ajusta la longitud de todas las secuencias a `MAX_LEN`. Las secuencias más cortas se rellenan con tokens `[PAD]` y las más largas se truncan.
    5.  **Máscara de Atención:** Genera una máscara binaria (`attention_mask`) que indica al modelo qué tokens son reales (1) y cuáles son de padding (0). El mecanismo de auto-atención de BERT usará esta máscara para ignorar los tokens de padding.
* **Resultado:** Se obtienen los `input_ids` y `attention_mask` como tensores numéricos, listos para ser la entrada del modelo BERT.

**Relevancia para BERT:** BERT no entiende texto directamente. Requiere estas secuencias numéricas (`input_ids`) de longitud fija y la `attention_mask` para procesar la información contextual correctamente y enfocar su atención en los tokens relevantes.
""")
st.code("""
# --- Tokenización ---
print(f"\\nCargando tokenizador para el modelo: {MODELO_BERT}")
tokenizer = BertTokenizerFast.from_pretrained(MODELO_BERT)

# Tokenizar los datasets de entrenamiento y validación
train_encodings = tokenizer(train_texts, truncation=True, padding='max_length', max_length=MAX_LEN)
val_encodings = tokenizer(val_texts, truncation=True, padding='max_length', max_length=MAX_LEN)
# train_encodings es un diccionario con 'input_ids', 'token_type_ids', 'attention_mask'
""", language="python")

st.subheader("5. Creación de Datasets y DataLoaders PyTorch")
st.markdown("""
Para integrar los datos con PyTorch, se utiliza una clase `Dataset` personalizada (`FrasesDataset`). Esta clase simplemente toma los diccionarios de encodings (resultado de la tokenización) y las listas de etiquetas numéricas, y define cómo obtener un ejemplo específico (`__getitem__`) convirtiéndolo en tensores de PyTorch.

Los `DataLoader`s se crean a partir de estos `Dataset`s. Su función es agrupar los ejemplos en lotes (`batch_size`), barajar los datos de entrenamiento (`shuffle=True`) para evitar que el modelo aprenda el orden, y cargar los datos de forma eficiente, especialmente útil cuando se trabaja con GPUs.

**Relevancia para BERT/PyTorch:** Este es el paso estándar para preparar los datos antes de alimentar un modelo PyTorch. El entrenamiento por lotes es crucial para manejar la memoria y mejorar el proceso de optimización.
""")
st.code("""
# --- Crear Clase de Dataset para PyTorch ---
class FrasesDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Obtiene los datos tokenizados para el índice idx
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Añade la etiqueta numérica (ya codificada)
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Crear instancias del Dataset
train_dataset = FrasesDataset(train_encodings, train_labels)
val_dataset = FrasesDataset(val_encodings, val_labels)

# --- Crear DataLoaders ---
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
""", language="python")

st.subheader("6. Carga del Modelo y Configuración de Optimización")
st.markdown("""
Se carga el modelo BETO pre-entrenado adaptado para clasificación de secuencias:
* `BertForSequenceClassification.from_pretrained(MODELO_BERT, num_labels=num_labels)`: Carga los pesos pre-entrenados de BETO y **añade automáticamente una capa lineal de clasificación** al final, conectada a la salida del token `[CLS]`. El parámetro `num_labels` asegura que esta capa tenga el número correcto de neuronas de salida (una por cada categoría de intención). La advertencia sobre pesos no inicializados ("Some weights were not initialized...") se refiere precisamente a esta nueva capa de clasificación, que se entrenará desde cero (o casi) durante el fine-tuning.
* El modelo se transfiere al dispositivo computacional adecuado (`cuda` o `cpu`).
* **Optimizador (`AdamW`)**: Se configura el optimizador AdamW, que calculará cómo actualizar los pesos del modelo basándose en los gradientes de la función de pérdida.
* **Planificador de Tasa de Aprendizaje (`get_linear_schedule_with_warmup`)**: Se define un scheduler para ajustar la tasa de aprendizaje durante el entrenamiento. Una práctica común es usar un "calentamiento" (warmup) inicial con tasa baja y luego disminuirla linealmente, lo que puede ayudar a que el modelo converja mejor.

**Relevancia para BERT:** Se instancia el modelo correcto para la tarea (`BertForSequenceClassification`). Cargar los pesos pre-entrenados es la esencia del *transfer learning*, aprovechando el conocimiento lingüístico general de BETO. El optimizador y el scheduler son cruciales para el proceso de aprendizaje durante el fine-tuning.
""")
st.code("""
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
# ...

# --- Cargar Modelo Pre-entrenado con Cabeza de Clasificación ---
print(f"\\nCargando modelo pre-entrenado: {MODELO_BERT}")
model = BertForSequenceClassification.from_pretrained(
    MODELO_BERT,
    num_labels=num_labels # Indica el número de intenciones a clasificar
)

# --- Configurar Dispositivo ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"\\nUsando dispositivo: {device}")

# --- Optimizador y Planificador ---
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
num_training_steps = len(train_loader) * EPOCHS
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)
""", language="python")

st.subheader("7. Bucle de Entrenamiento y Validación")
st.markdown("""
Este es el ciclo donde el modelo aprende. Itera a través de las `EPOCHS`. Cada época consiste en una fase de entrenamiento y una de validación:

1.  **Entrenamiento (`model.train()`):**
    * El modelo procesa los datos del `train_loader` lote por lote.
    * Para cada lote:
        * Los datos (`input_ids`, `attention_mask`, `labels`) se mueven al `device`.
        * **Forward Pass:** Se alimentan los `input_ids` y `attention_mask` al modelo (`outputs = model(...)`). BERT genera embeddings contextuales internamente y la cabeza de clasificación produce `logits`. Al pasar también las `labels` verdaderas, el modelo calcula automáticamente la **pérdida (loss)** (ej: Cross-Entropy Loss).
        * **Backward Pass:** Se calculan los gradientes de la pérdida respecto a todos los parámetros entrenables del modelo (`loss.backward()`).
        * **Actualización:** El optimizador ajusta los parámetros del modelo en la dirección que reduce la pérdida (`optimizer.step()`).
        * Se actualiza la tasa de aprendizaje (`lr_scheduler.step()`).
    * Se calcula la pérdida promedio de entrenamiento para la época.
2.  **Validación (`model.eval()`, `torch.no_grad()`):**
    * El modelo procesa los datos del `val_loader` sin calcular gradientes y con capas como Dropout desactivadas.
    * Para cada lote, se obtienen los `logits`.
    * Se calcula la clase predicha tomando el índice del logit más alto (`torch.argmax`).
    * Se acumulan las predicciones y las etiquetas verdaderas.
    * Al final de la época, se calculan métricas de rendimiento (Accuracy, F1-Score ponderado) sobre todo el conjunto de validación.
3.  **Guardado del Mejor Modelo:** Se compara la métrica de validación (ej: Accuracy) con la mejor obtenida hasta el momento. Si mejora, se guarda el estado actual del modelo y del tokenizador en `OUTPUT_DIR`. Esto asegura que se conserve la versión del modelo que mejor funcionó en datos no vistos.

**Relevancia para BERT:** Este es el proceso iterativo de **fine-tuning**. El modelo BERT, con su cabeza de clasificación, aprende a asociar patrones en las frases (capturados a través de los embeddings contextuales del token `[CLS]`) con las categorías de intención correctas. La validación previene el sobreajuste al dataset de entrenamiento. Los **Embeddings** son el corazón de la representación interna de BERT.
""")
st.code("""
print("\\n--- Iniciando Entrenamiento ---")
best_val_accuracy = 0.0

for epoch in range(EPOCHS):
    # --- Fase de Entrenamiento ---
    model.train()
    total_train_loss = 0
    progress_bar_train = tqdm(train_loader, desc=f"Entrenando Época {epoch+1}", leave=False)
    for batch in progress_bar_train:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        progress_bar_train.set_postfix({'loss': loss.item()})
    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Pérdida media de entrenamiento: {avg_train_loss:.4f}")

    # --- Fase de Validación ---
    model.eval()
    total_val_loss = 0
    all_preds = []
    all_labels = []
    progress_bar_val = tqdm(val_loader, desc=f"Validando Época {epoch+1}", leave=False)
    with torch.no_grad():
        for batch in progress_bar_val:
            # ... (mover batch a device) ...
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_val_loss += loss.item()
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            progress_bar_val.set_postfix({'loss': loss.item()})
    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"Pérdida media de validación: {avg_val_loss:.4f}")
    print(f"Accuracy de validación: {val_accuracy:.4f}")
    print(f"F1-Score (Weighted) de validación: {val_f1:.4f}")

    # --- Guardar el mejor modelo ---
    if val_accuracy > best_val_accuracy:
        print(f"¡Mejora en Accuracy! Guardando modelo...")
        best_val_accuracy = val_accuracy
        model.save_pretrained(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR) # Guardar tokenizer junto al modelo
        # El mapeo ya se guardó antes
    # ...

print("\\n--- Entrenamiento Completado ---")
""", language="python")


st.subheader("8. Carga y Predicción (Inferencia)")
st.markdown("""
Finalmente, se muestra cómo cargar el modelo afinado desde el `OUTPUT_DIR` y usarlo para predecir la intención de frases nuevas:
1.  Se cargan los mapeos `id2label`/`label2id` del archivo JSON.
2.  Se cargan el modelo (`BertForSequenceClassification.from_pretrained`) y el tokenizador (`BertTokenizerFast.from_pretrained`) usando la ruta del directorio donde se guardó el mejor modelo.
3.  El modelo se configura en modo evaluación (`.eval()`).
4.  La función `predecir_categoria` realiza la inferencia:
    * **Tokeniza** la frase de entrada.
    * Ejecuta el **forward pass** en el modelo (sin calcular gradientes).
    * Obtiene los `logits` de salida de la cabeza de clasificación.
    * Encuentra el **ID de la clase predicha** (`torch.argmax`).
    * Usa el mapeo `id2label` para **convertir el ID a la etiqueta de intención** textual.
5.  Un bucle simple permite probar el modelo interactivamente.

**Relevancia para BERT:** Este bloque simula exactamente lo que hace la función `predict_intent` del chatbot: toma texto, lo tokeniza, lo pasa por el modelo BERT afinado para obtener una predicción de intención, y devuelve la intención como texto legible.
""")
st.code("""
import torch
from transformers import BertForSequenceClassification, BertTokenizerFast
import json
import os

# --- Configuración ---
MODELO_GUARDADO_DIR = 'drive/MyDrive/PLN_Segundo_Momento/modelo_clasificador_frases'
MAX_LEN = 128

# --- Cargar Mapeo, Modelo y Tokenizador Guardados ---
try:
    map_file_path = os.path.join(MODELO_GUARDADO_DIR, 'label_mappings.json')
    with open(map_file_path, 'r', encoding='utf-8') as f:
        mappings = json.load(f)
    id2label = {int(k): v for k, v in mappings['id2label'].items()} # Importante convertir keys a int
    label2id = mappings['label2id']

    model_cargado = BertForSequenceClassification.from_pretrained(MODELO_GUARDADO_DIR)
    tokenizer_cargado = BertTokenizerFast.from_pretrained(MODELO_GUARDADO_DIR)
except Exception as e:
    # ... (Manejo de errores) ...

# --- Configurar Dispositivo y Modo Eval ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_cargado.to(device)
model_cargado.eval() # Modo evaluación

# --- Función de Predicción ---
def predecir_categoria(texto, modelo, tokenizer, device, id_to_label_map, max_len):
    # ... (Tokenizar entrada) ...
    inputs = tokenizer(texto, return_tensors='pt', max_length=max_len, padding='max_length', truncation=True)
    # ... (Mover inputs a device) ...
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    # ... (Inferencia con torch.no_grad()) ...
    with torch.no_grad():
        outputs = modelo(input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    # ... (Obtener ID predicho con argmax) ...
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    # ... (Mapear ID a etiqueta de texto) ...
    categoria_predicha = id_to_label_map.get(predicted_class_id, "Categoría Desconocida")
    return categoria_predicha

# --- Bucle de Chat Simple ---
# ... (Bucle while que usa predecir_categoria) ...
""", language="python")


st.header("Conclusión Final (Parte 1: Clasificación de Intención)")

st.success("""
La primera parte del notebook `PLN2Momento (1).ipynb` demuestra el proceso completo de **fine-tuning de un modelo BERT (BETO) para clasificar la intención de frases en español**. Se utiliza un dataset específico (`dataset_completo.csv`) para adaptar el conocimiento general del modelo pre-entrenado a las categorías de intención relevantes para el chatbot (Saludo, Insulto, Respuesta Nombre, etc.). El proceso incluye la preparación de datos (codificación de etiquetas, tokenización), el entrenamiento iterativo con validación, y el guardado del modelo final (`modelo_clasificador_frases`). Este modelo afinado es el componente que permite al chatbot realizar el primer nivel de comprensión del lenguaje del usuario, identificando el propósito de sus mensajes para guiar la conversación.
""")
