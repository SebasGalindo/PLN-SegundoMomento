# -*- coding: utf-8 -*-
import json
import os
import random
import re
import joblib
import spacy
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification
from pathlib import Path
import time
import yfinance as yf
import datetime

# --- 1. Configuración de Rutas y Constantes ---
BASE_DIR = Path(__file__).resolve().parent.parent # Asume que este script está en la misma carpeta que Data
DATA_DIR = BASE_DIR / "Data"
INTENT_MODEL_DIR = DATA_DIR / "modelo_clasificador_frases"
NER_MODEL_DIR = DATA_DIR / "modelo_ner_frases"
LGBM_MODEL_PATH = DATA_DIR / "model_bundle_nivel_economico.joblib"
QUESTIONS_ANSWERS_PATH = DATA_DIR / "questions_and_answers.json"
CATEGORIES_PATH = DATA_DIR / "Categorias-Empresa.txt"

# Modelos y Constantes
MODELO_INTENCION = str(INTENT_MODEL_DIR)
MODELO_NER = str(NER_MODEL_DIR)
MODELO_SPACY = "es_core_news_lg"
MAX_LEN_BERT = 128
DEFAULT_CURRENCY = "COP"
SIMILARITY_THRESHOLD = 0.4

CAMPOS_REQUERIDOS = [
    "nombre_empresa", "area_categoria", "numero_empleados",
    "ingresos_o_activos", "valor_cartera", "valor_deudas"
]
CAMPO_A_NER_LABEL = {
    "nombre_empresa": ["NOMBRE_EMPRESA"],
    "area_categoria": ["CATEGORIA_EMPRESA", "SECTOR"],
    "numero_empleados": ["NUM_EMPLEADOS"],
    "ingresos_o_activos": ["VALOR_GANANCIAS", "VALOR_ACTIVOS"],
    "valor_cartera": ["VALOR_CARTERA"],
    "valor_deudas": ["VALOR_DEUDAS"],
}
# Diccionario para almacenar listas de preguntas por campo
CAMPO_A_LISTA_PREGUNTAS = {}

# Palabras clave/lemas para confirmación/negación (expandido)
LEMAS_POSITIVOS = {"sí", "si", "correcto", "verdadero", "afirmativo", "confirmar", "ok", "dale", "valer", "exacto", "así", "ser", "acuerdo", "proceder", "ajá", "efectivamente", "claro"}
LEMAS_NEGATIVOS = {"no", "incorrecto", "falso", "negativo", "error", "mal", "nunca", "jamás", "tampoco"}
# Palabras que directamente niegan (sin lematizar)
PALABRAS_NEGACION_DIRECTA = {"no", "incorrecto", "falso", "negativo"}

# --- 2. Carga de Recursos ---
print("--- Cargando Recursos del Chatbot ---")

# 2.1 Cargar JSON y extraer datos
try:
    with open(QUESTIONS_ANSWERS_PATH, 'r', encoding='utf-8') as f:
        q_and_a_full = json.load(f)

    respuestas_saludo = q_and_a_full.get('respuestas_saludo', ["¡Hola nombre! ¿Nombre de la empresa?"])
    respuestas_insulto = q_and_a_full.get('respuestas_insulto', ["Por favor, se respetuoso."])
    respuestas_pregunta_personal = q_and_a_full.get('respuestas_pregunta_personal', ["No puedo responder eso."])
    respuestas_acuse_recibo = q_and_a_full.get('respuestas_acuse_recibo', ["Ok."])
    frases_confirmacion = q_and_a_full.get('frases_confirmacion', ["¿Es {valor} correcto para {campo}?"])
    respuestas_manejo_correccion = q_and_a_full.get('respuestas_manejo_correccion', ["Ok, ¿cuál es el valor correcto?"])
    respuestas_despedida = q_and_a_full.get('respuestas_despedida', ["¡Adiós!"])
    plantillas_resultado_final = q_and_a_full.get('plantillas_resultado_final', {})
    acronimos_sociedades = q_and_a_full.get('acronimos_sociedades', []) # Cargar acrónimos

    info_chatbot = q_and_a_full.get('informacion_chatbot', {})
    descripcion_general = info_chatbot.get('descripcion_general', "Soy un chatbot financiero.")
    terminos_economicos_list = info_chatbot.get('terminos_economicos', [])
    terminos_economicos = {item['termino'].lower(): item['definicion'] for item in terminos_economicos_list if 'termino' in item and 'definicion' in item}
    explicacion_proceso_campos = info_chatbot.get('explicacion_proceso_campos', {})

    # Poblar CAMPO_A_LISTA_PREGUNTAS (Ahora almacena la lista)
    for campo, data in explicacion_proceso_campos.items():
        if 'pregunta_asociada' in data and isinstance(data['pregunta_asociada'], list):
             CAMPO_A_LISTA_PREGUNTAS[campo] = data['pregunta_asociada']
        elif 'pregunta_asociada' in data: # Si por error sigue siendo string
             CAMPO_A_LISTA_PREGUNTAS[campo] = [data['pregunta_asociada']]


    print("JSON de preguntas y respuestas cargado.")
except FileNotFoundError:
    print(f"Error Crítico: No se encontró el archivo JSON en {QUESTIONS_ANSWERS_PATH}")
    exit()
except (json.JSONDecodeError, KeyError, TypeError) as e:
    print(f"Error Crítico: El archivo JSON está mal formado o falta estructura clave: {e}")
    exit()

# 2.2 Cargar Modelo y Tokenizador de Clasificación de Intención
print("Cargando modelo de clasificación de intención...")
# (Código de carga sin cambios)
try:
    intent_tokenizer = AutoTokenizer.from_pretrained(MODELO_INTENCION)
    intent_model = AutoModelForSequenceClassification.from_pretrained(MODELO_INTENCION)
    intent_map_path = INTENT_MODEL_DIR / 'label_mappings.json'
    with open(intent_map_path, 'r', encoding='utf-8') as f:
        intent_mappings = json.load(f)
    intent_id2label = {int(k): v for k, v in intent_mappings['id2label'].items()}
    intent_label2id = intent_mappings['label2id']
    print("Modelo de Intención cargado.")
except Exception as e:
    print(f"Error Crítico cargando modelo de Intención desde {INTENT_MODEL_DIR}: {e}")
    exit()

# 2.3 Cargar Modelo y Tokenizador NER
print("Cargando modelo NER...")
# (Código de carga sin cambios)
try:
    ner_tokenizer = AutoTokenizer.from_pretrained(MODELO_NER)
    ner_model = AutoModelForTokenClassification.from_pretrained(MODELO_NER)
    ner_map_path = NER_MODEL_DIR / 'ner_label_mappings.json'
    with open(ner_map_path, 'r', encoding='utf-8') as f:
        ner_mappings = json.load(f)
    ner_id2label = {int(k): v for k, v in ner_mappings['id2label'].items()}
    ner_label2id = ner_mappings['label2id']
    base_ner_labels = set()
    for label in ner_id2label.values():
        if label != 'O':
            base_ner_labels.add(label[2:])
    print("Modelo NER cargado.")
except Exception as e:
    print(f"Error Crítico cargando modelo NER desde {NER_MODEL_DIR}: {e}")
    exit()

# 2.4 Cargar Modelo LightGBM
print("Cargando modelo de clasificación de nivel económico (LightGBM)...")
# (Código de carga sin cambios)
try:
    lgbm_bundle = joblib.load(LGBM_MODEL_PATH)
    lgbm_model = lgbm_bundle['model']
    lgbm_label_encoder = lgbm_bundle['label_encoder']
    lgbm_features = lgbm_bundle['features']
    print("Modelo LightGBM cargado.")
    print(f"  -> Features esperadas por LightGBM: {lgbm_features}")
except FileNotFoundError:
     print(f"Error Crítico: No se encontró el archivo del modelo LightGBM en {LGBM_MODEL_PATH}")
     exit()
except Exception as e:
     print(f"Error Crítico cargando el modelo LightGBM: {e}")
     exit()

# 2.5 Cargar Modelo spaCy y Categorías con Embeddings
print(f"Cargando modelo spaCy '{MODELO_SPACY}'...")
# (Código de carga sin cambios, ya usa spaCy)
try:
    nlp = spacy.load(MODELO_SPACY)
    print("Modelo spaCy cargado.")
    categorias_df = pd.read_csv(CATEGORIES_PATH, header=None, names=['Area', 'Sector'], skipinitialspace=True, encoding='utf-8')
    categorias_df.dropna(inplace=True)
    categorias_df = categorias_df[categorias_df['Sector'].str.upper() != 'N/A']
    categorias_empresa = {}
    print("Vectorizando categorías de empresa...")
    for index, row in categorias_df.iterrows():
        area = row['Area'].strip()
        sector = row['Sector'].strip()
        doc = nlp(area.lower())
        if doc.has_vector and doc.vector_norm:
            categorias_empresa[area] = {'vector': doc.vector / doc.vector_norm, 'sector': sector}
        else:
            print(f"  Advertencia: No se pudo obtener vector para '{area}'. Se omitirá.")
    print(f"{len(categorias_empresa)} categorías vectorizadas.")
    if not categorias_empresa:
        print("Error Crítico: No se pudieron vectorizar categorías.")
        exit()
except OSError:
     print(f"Error Crítico: Modelo spaCy '{MODELO_SPACY}' no encontrado. Descárgalo con:")
     print(f"python -m spacy download {MODELO_SPACY}")
     exit()
except FileNotFoundError:
     print(f"Error Crítico: No se encontró el archivo de categorías en {CATEGORIES_PATH}")
     exit()
except Exception as e:
     print(f"Error Crítico cargando spaCy o categorías: {e}")
     exit()

# 2.6 Configurar Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
intent_model.to(device)
intent_model.eval()
ner_model.to(device)
ner_model.eval()
print(f"\nModelos Transformers listos en dispositivo: {device}")

print("\n--- Todos los recursos cargados exitosamente ---")


print(f"Intentando obtener tasas desde Yahoo Finance (aprox. {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
print("-" * 40)

# --- Tickers para Yahoo Finance ---
ticker_usd_cop = 'USDCOP=X'
ticker_eur_cop = 'EURCOP=X'

tasas = {}

# --- Obtener tasa USD/COP ---
try:
    print(f"Buscando ticker: {ticker_usd_cop}...")
    ticker_obj = yf.Ticker(ticker_usd_cop)

    hist = ticker_obj.history(period='1d', interval='5m') # Último día, intervalos de 5 min
    if not hist.empty:
        tasa_actual = hist['Close'].iloc[-1]
        tasas['USD_COP'] = tasa_actual
        print(f"Tasa USD/COP encontrada (historial reciente): {tasa_actual:,.2f}")
    else:
        info = ticker_obj.info
        if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
             tasa_actual = info['regularMarketPrice']
             tasas['USD_COP'] = tasa_actual
             print(f"Tasa USD/COP encontrada (info general): {tasa_actual:,.2f}")
        elif 'bid' in info and info['bid'] is not None: 
             tasa_actual = info['bid']
             tasas['USD_COP'] = tasa_actual
             print(f"Tasa USD/COP encontrada (bid): {tasa_actual:,.2f}")
        else:
             print(f"No se pudo obtener la tasa actual para {ticker_usd_cop} desde Yahoo Finance.")
             print("Ticker info:", info) 

except Exception as e:
    print(f"Error obteniendo datos para {ticker_usd_cop}: {e}")

print("-" * 40)

# --- Obtener tasa EUR/COP ---
try:
    print(f"Buscando ticker: {ticker_eur_cop}...")
    ticker_obj = yf.Ticker(ticker_eur_cop)

    hist = ticker_obj.history(period='1d', interval='5m')
    if not hist.empty:
        tasa_actual = hist['Close'].iloc[-1]
        tasas['EUR_COP'] = tasa_actual
        print(f"Tasa EUR/COP encontrada (historial reciente): {tasa_actual:,.2f}")
    else:
        info = ticker_obj.info
        if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
             tasa_actual = info['regularMarketPrice']
             tasas['EUR_COP'] = tasa_actual
             print(f"Tasa EUR/COP encontrada (info general): {tasa_actual:,.2f}")
        elif 'bid' in info and info['bid'] is not None:
             tasa_actual = info['bid']
             tasas['EUR_COP'] = tasa_actual
             print(f"Tasa EUR/COP encontrada (bid): {tasa_actual:,.2f}")
        else:
             print(f"No se pudo obtener la tasa actual para {ticker_eur_cop} desde Yahoo Finance.")
             print("Ticker info:", info)


except Exception as e:
    print(f"Error obteniendo datos para {ticker_eur_cop}: {e}")

print("-" * 40)
print("Resumen:")
if 'USD_COP' in tasas:
    print(f"1 USD = {tasas['USD_COP']:,.2f} COP")
else:
    print("No se obtuvo la tasa USD/COP.")

if 'EUR_COP' in tasas:
    print(f"1 EUR = {tasas['EUR_COP']:,.2f} COP")
else:
    print("No se obtuvo la tasa EUR/COP.")

# --- 3. Funciones Auxiliares (Modificadas/Nuevas) ---

print("\nINFO: La Tokenización es realizada internamente por los modelos BERT y spaCy.")

def predict_intent(text):
    inputs = intent_tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LEN_BERT)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = intent_model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    return intent_id2label.get(predicted_class_id, "Intencion Desconocida")

def predict_ner(text):
    """Predice entidades NER usando el modelo BETO #2. Devuelve lista de entidades."""
    inputs = ner_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LEN_BERT, return_offsets_mapping=True)
    offset_mapping = inputs.pop("offset_mapping").cpu().squeeze().tolist()
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = ner_model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1).cpu().squeeze().tolist()
    entities = []
    current_entity_tokens = []
    current_entity_label = None
    start_offset = -1
    current_entity_end_offset = -1
    for i, pred_id in enumerate(predictions):
        start_char, end_char = offset_mapping[i]
        if start_char == end_char:
            if current_entity_tokens: entities.append({"text": text[start_offset:current_entity_end_offset],"label": current_entity_label,"start": start_offset,"end": current_entity_end_offset}); current_entity_tokens = []; current_entity_label = None; start_offset = -1
            continue
        label_name = ner_id2label.get(pred_id, "O")
        if label_name.startswith("B-"):
            if current_entity_tokens: entities.append({"text": text[start_offset:current_entity_end_offset],"label": current_entity_label,"start": start_offset,"end": current_entity_end_offset})
            current_entity_tokens = [(start_char, end_char)]; current_entity_label = label_name[2:]; start_offset = start_char; current_entity_end_offset = end_char
        elif label_name.startswith("I-"):
            if current_entity_tokens and current_entity_label == label_name[2:]: current_entity_tokens.append((start_char, end_char)); current_entity_end_offset = end_char
            elif current_entity_tokens: entities.append({"text": text[start_offset:current_entity_end_offset],"label": current_entity_label,"start": start_offset,"end": current_entity_end_offset}); current_entity_tokens = []; current_entity_label = None; start_offset = -1
        elif label_name == "O":
            if current_entity_tokens: entities.append({"text": text[start_offset:current_entity_end_offset],"label": current_entity_label,"start": start_offset,"end": current_entity_end_offset}); current_entity_tokens = []; current_entity_label = None; start_offset = -1
    if current_entity_tokens: entities.append({"text": text[start_offset:current_entity_end_offset],"label": current_entity_label,"start": start_offset,"end": current_entity_end_offset})
    return entities

multiplicadores_spacy = {'mil': 1000, 'millon': 1000000, 'millón': 1000000, 'millones': 1000000}
numeros_texto_spacy = {
    'un': 1, 'uno': 1, 'dos': 2, 'tres': 3, 'tre':3, 'cuatro': 4, 'cinco': 5,
    'seis': 6, 'siete': 7, 'ocho': 8, 'nueve': 9, 'diez': 10,
    'once': 11, 'doce': 12, 'trece': 13, 'catorce': 14, 'quince': 15,
    'dieciseis': 16, 'diecisiete': 17, 'dieciocho': 18, 'diecinueve': 19,
    'veinte': 20, 'veinti': 20, 'veintiun': 21, 'veintiuno': 21, 'veintidos': 22, 'veintitres': 23,
    'veinticuatro': 24, 'veinticinco': 25, 'veintiseis': 26, 'veintisiete': 27,
    'veintiocho': 28, 'veintinueve': 29,
    'treinta': 30, 'cuarenta': 40, 'cincuenta': 50, 'sesenta': 60,
    'setenta': 70, 'ochenta': 80, 'noventa': 90,
    'cien': 100, 'ciento': 100, 'cientos': 100, # 'cientos' actúa como base 100 y multiplicador parcial
    'doscientos': 200, 'trescientos': 300, 'trecientos': 300, # Error común
    'cuatrocientos': 400, 'quinientos': 500, 'seiscientos': 600,
    'setecientos': 700, 'ochocientos': 800, 'novecientos': 900
}
conectores_spacy = {'y', 'de'}

def parse_numero(texto):
    """
    Intenta convertir texto a número usando spaCy para tokenizar/lematizar
    y reglas para combinar números y multiplicadores. Extrae moneda.
    """
    if not texto or not isinstance(texto, str):
        return None, None

    texto_original = texto
    print(f"DEBUG parse_numero (spaCy): Recibido '{texto_original}'")

    # 1. Limpieza inicial y extracción de moneda 
    texto_limpio = texto.lower().strip()
    texto_limpio = re.sub(r'[$\']', '', texto_limpio)
    texto_limpio = re.sub(r'\s+', ' ', texto_limpio)

    monedas_map = {'pesos': 'COP', 'peso': 'COP', 'dolar': 'USD', 'dólar': 'USD', 'dolares': 'USD', 'dólares': 'USD', 'eur': 'EUR', 'euro': 'EUR', 'euros': 'EUR', 'cop': 'COP', 'usd': 'USD'}
    moneda = DEFAULT_CURRENCY
    texto_sin_moneda = texto_limpio
    moneda_detectada_str = None

    for palabra, codigo in monedas_map.items():
        pattern = r'\b' + re.escape(palabra) + r'\b'
        match = re.search(pattern, texto_limpio)
        if match:
            moneda = codigo; moneda_detectada_str = match.group(0)
            texto_sin_moneda = re.sub(pattern, '', texto_sin_moneda).strip()
            print(f"DEBUG parse_numero: Moneda detectada: {moneda} (palabra: '{moneda_detectada_str}')")
            break

    # 2. Eliminar palabras irrelevantes/relleno 
    relleno = ['unos', 'aproximadamente', 'cerca de', 'alrededor de', 'como', 'un total de', 'más o menos', 'de']
    for palabra_relleno in relleno:
        texto_sin_moneda = re.sub(r'\b' + re.escape(palabra_relleno) + r'\b', '', texto_sin_moneda).strip()
    texto_sin_moneda = re.sub(r'\s+', ' ', texto_sin_moneda).strip()
    print(f"DEBUG parse_numero: Texto para procesar: '{texto_sin_moneda}'")

    doc = nlp(texto_sin_moneda)
    total_valor = 0.0
    valor_segmento = 0.0 
    segmento_con_digito = False # Flag para saber si el segmento actual contiene dígitos

    for token in doc:
        print(f"  Procesando Token: '{token.text}' (Lema: {token.lemma_}, like_num: {token.like_num})")

        if token.like_num:
            try:
                # Convertir a float, reemplazando ',' por '.' y eliminando espacios
                num_text = token.text.replace(',', '.').replace(' ', '')
                # si hay mas de un . eliminarlos
                if num_text.count('.') > 1:
                    num_text = num_text.replace('.', '')
                num_digit = float(num_text  )
                # Si ya había algo en el segmento (ej, de texto), lo sobreescribe el dígito
                valor_segmento = num_digit
                segmento_con_digito = True # Marcar que este segmento usa dígitos
                print(f"    -> Detectado como dígito: {num_digit}. valor_segmento={valor_segmento}")
            except ValueError:
                print(f"    -> 'like_num' pero no se pudo convertir a float: '{token.text}'")
                segmento_con_digito = False # Resetear flag si falla la conversión
                
        # Si es una palabra numérica conocida
        if token.lemma_ in numeros_texto_spacy and not segmento_con_digito:
             # Caso especial "ciento(s)" - puede multiplicar lo anterior o ser base 100
             if token.lemma_ in ['ciento', 'cien', 'cientos']:
                 if valor_segmento > 0: 
                     valor_segmento *= 100
                 else: 
                     valor_segmento += 100
             else:
                 valor_segmento += numeros_texto_spacy[token.lemma_]
             print(f"    -> Detectado como palabra numérica: {numeros_texto_spacy[token.lemma_]}. valor_segmento={valor_segmento}")

        if token.lemma_ in multiplicadores_spacy:
            if valor_segmento == 0:
                valor_segmento = 1
                print(f"    -> Multiplicador sin número previo, usando 1.")

            valor_aplicado = valor_segmento * multiplicadores_spacy[token.lemma_]
            total_valor += valor_aplicado
            print(f"    -> Aplicando multiplicador '{token.lemma_}'. Sumando {valor_aplicado}. total_valor={total_valor}")
            valor_segmento = 0
            segmento_con_digito = False


        # Ignorar conectores y otras palabras
        if token.lemma_ not in conectores_spacy and not (token.lemma_ in numeros_texto_spacy or token.lemma_ in multiplicadores_spacy or token.like_num):
             print(f"    -> Palabra '{token.text}' (lema: {token.lemma_}) ignorada.")
             if valor_segmento > 0:
                 print(f"    -> Palabra desconocida encontrada, sumando segmento pendiente: {valor_segmento}")
                 total_valor += valor_segmento
                 valor_segmento = 0
                 segmento_con_digito = False


    if valor_segmento > 0:
        total_valor += valor_segmento
        print(f"DEBUG parse_numero: Sumando segmento final remanente: {valor_segmento}. total_valor={total_valor}")

    if total_valor > 0:
        print(f"DEBUG parse_numero: Valor final extraído: {total_valor}")
        return total_valor, moneda
    else:
        try:
            num_str_directo = texto_sin_moneda.replace('.', '').replace(',', '.').replace(' ', '')
            if num_str_directo:
                 valor = float(num_str_directo)
                 print(f"DEBUG parse_numero: Fallback final - conversión directa funcionó: {valor}")
                 return valor, moneda
        except ValueError:
             print(f"DEBUG parse_numero: No se pudo extraer un valor numérico de '{texto_original}' por ningún método.")
             return None, None # Falla total

def encontrar_mejor_categoria(texto_categoria): # Ya no necesita el parámetro umbral aquí
    """
    Encuentra la categoría conocida más similar usando embeddings de spaCy.
    Devuelve SIEMPRE la mejor categoría encontrada, su sector y la similitud.
    """
    print(f"\n--- Buscando similitud para: '{texto_categoria}' (Embeddings spaCy) ---")
    doc_usuario = nlp(texto_categoria.lower())
    if not doc_usuario.has_vector or not doc_usuario.vector_norm:
        print("  Advertencia: No se pudo obtener vector para la entrada.")
        return None, None, 0.0

    vector_usuario_norm = doc_usuario.vector / doc_usuario.vector_norm

    mejor_cat_encontrada = None
    mejor_sector_encontrado = None
    max_sim = -1.0 

    for cat_original, data in categorias_empresa.items():
        similitud = np.dot(vector_usuario_norm, data['vector'])
        print(f"  Comparando con '{cat_original}': Similitud = {similitud:.4f}")
        if similitud > max_sim:
            max_sim = similitud
            mejor_cat_encontrada = cat_original
            mejor_sector_encontrado = data['sector']

    print(f"----------------------------------------------------")
    print(f"DEBUG: Mejor coincidencia preliminar: '{mejor_cat_encontrada}' (Sector: {mejor_sector_encontrado}, Sim: {max_sim:.4f})")
    return mejor_cat_encontrada, mejor_sector_encontrado, max_sim

def get_next_question(contexto):
    for campo in CAMPOS_REQUERIDOS:
        if campo == 'ingresos_o_activos':
             if contexto.get('valor_ingresos') is None and contexto.get('valor_activos') is None: return 'valor_ingresos'
        elif contexto.get(campo) is None: return campo
    return None

# MODIFICADO: Selecciona pregunta aleatoria
def get_formatted_question(campo_key):
    """Obtiene aleatoriamente el texto de la pregunta para una clave de campo desde el JSON."""
    lista_preguntas = CAMPO_A_LISTA_PREGUNTAS.get(campo_key)
    if lista_preguntas and isinstance(lista_preguntas, list):
        pregunta = random.choice(lista_preguntas)
    else: 
        pregunta = f"¿Podrías darme información sobre {campo_key.replace('_', ' ')}?"
        print(f"ADVERTENCIA: No se encontró lista de preguntas para '{campo_key}', usando fallback.")

    # Añadir ejemplos si existen
    if campo_key in explicacion_proceso_campos:
         ejemplo = explicacion_proceso_campos[campo_key].get('ejemplo_respuesta')
         if ejemplo:
              pregunta += f" (Ej: {ejemplo})" # Añadir ejemplo al final
    return pregunta

def format_greeting(template, name=None):
    if name: return template.replace("nombre", name)
    else: return re.sub(r'\s?nombre', '', template)

def check_confirmation(text):
    """Clasifica la respuesta de confirmación usando Lematización y POS Tagging."""
    print("\n--- Analizando Confirmación/Negación (Lema/POS) ---")
    doc = nlp(text.lower()) 
    has_negation = False
    has_affirmation = False

    for token in doc:
        print(f"  Token: {token.text}, Lema: {token.lemma_}, POS: {token.pos_}")
        if token.lemma_ in LEMAS_NEGATIVOS or token.text in PALABRAS_NEGACION_DIRECTA:
             is_double_negation = False
             if token.i > 0 and doc[token.i-1].lemma_ == "ser":
                 if token.i + 1 < len(doc) and doc[token.i+1].lemma_ in LEMAS_NEGATIVOS: 
                      is_double_negation = True

             if not is_double_negation:
                 print("  -> Negación detectada.")
                 has_negation = True
                 break 

        if token.lemma_ in LEMAS_POSITIVOS:
            print("  -> Afirmación detectada.")
            has_affirmation = True

    print("-------------------------------------------------")

    if has_negation:
        return 'no'
    elif has_affirmation:
        return 'yes'
    else:
        print("  -> Respuesta no clasificada como 'sí' o 'no' claros.")
        return 'unclear'

def convert_to_COP(valor, moneda):
    """Convierte el valor a COP usando tasas de cambio ficticias (ejemplo)."""
    tasas_cambio = {
        'USD': tasas["USD_COP"] if tasas["USD_COP"] is not None else 4300, 
        'EUR': tasas['EUR_COP'] if tasas['EUR_COP'] is not None else 4900,
    }
    
    if moneda == 'COP':
        return valor
    elif moneda in tasas_cambio.keys():
        tasa = tasas_cambio[moneda]
        valor_convertido = valor * tasa
        print(f"DEBUG: Convertido {valor} {moneda} a {valor_convertido:.2f} COP usando tasa {tasa:.2f}.")
        return valor_convertido

def format_data_for_lgbm(contexto):
    print("\n--- Formateando datos para LightGBM ---"); print(f"Contexto recibido: {contexto}")
    try:
        valor_activo_o_ingreso = 0
        if contexto.get('valor_activos') is not None: valor_activo_o_ingreso = contexto['valor_activos']
        elif contexto.get('valor_ingresos') is not None: valor_activo_o_ingreso = contexto['valor_ingresos']; print("  ADVERTENCIA: Usando valor_ingresos en lugar de valor_activos para el modelo.")
        if contexto.get('moneda_ingresos') is not None: 
            moneda_ingresos = contexto['moneda_ingresos']
        elif contexto.get('moneda_activos') is not None: 
            moneda_ingresos = contexto['moneda_activos']
        else: moneda_ingresos = DEFAULT_CURRENCY; print("  ADVERTENCIA: Usando moneda por defecto (COP) para el modelo.")
        
        data = {
            'Sector': [contexto.get('sector', 'Desconocido')],
            'Area': [contexto.get('area_categoria', 'Desconocida')],
            'Numero Empleados': [contexto.get('numero_empleados', 0)],
            'Activos (COP)': [convert_to_COP(valor_activo_o_ingreso, moneda_ingresos)],
            'Cartera (COP)': [convert_to_COP(contexto.get('valor_cartera', 0), contexto.get('moneda_cartera', DEFAULT_CURRENCY))],
            'Deudas (COP)': [convert_to_COP(contexto.get('valor_deudas', 0), contexto.get('moneda_deudas', DEFAULT_CURRENCY))],
            }
        df_row = pd.DataFrame.from_dict(data); print(f"DataFrame inicial para LGBM:\n{df_row}")
        for col in lgbm_features:
             if col not in df_row.columns: print(f"  ERROR: Falta la columna '{col}' esperada por LGBM."); return None
             if df_row[col].dtype.name in ['object', 'category']: df_row[col] = df_row[col].astype('category')
             else: df_row[col] = pd.to_numeric(df_row[col], errors='coerce').fillna(0)
        df_row = df_row[lgbm_features]; print(f"DataFrame final formateado para LGBM:\n{df_row}"); print(f"Tipos de datos finales: \n{df_row.dtypes}"); return df_row
    except Exception as e: print(f"Error formateando datos para LightGBM: {e}"); return None

def predict_lgbm(data_row):
    try:
        prediction_encoded = lgbm_model.predict(data_row)[0]
        prediction_label = lgbm_label_encoder.inverse_transform([prediction_encoded])[0]
        return prediction_label
    except ValueError as ve: print(f"Error de Valor durante predicción LightGBM: {ve}"); print("  Posible causa: Categoría no vista durante el entrenamiento."); print(f"  Datos de entrada: \n{data_row}"); return "Error: Categoría Desconocida"
    except Exception as e: print(f"Error durante la predicción LightGBM: {e}"); return "Error en Predicción"

def extract_name_from_greeting(text):
    """Intenta extraer un nombre propio del saludo inicial usando varios patrones."""
    # Patrones Regex (sensibles a mayúsculas al inicio de nombres)
    # Prioridad a patrones más explícitos
    patterns = [
        r'(me llamo|mi nombre es|soy)\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*)', # "Me llamo Juan", "Soy Ana Pérez"
        r'(buenos días|buenas tardes|buenas noches|hola|hey|qué tal)[,]?\s*(?:soy|me llamo)?\s*([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)*)', # "Buenos días, soy Carlos", "Hola, María"
        r'([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)\s+(?:por aquí|habla|le saluda)', # "Sebastian por aquí"
        # Patrón más general (menos prioritario, podría capturar otras cosas)
        r'^\s*([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)\b' # Un nombre al inicio de la frase
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            # El nombre suele estar en el último grupo de captura
            nombre_potencial = match.groups()[-1].strip()
            # Evitar capturar palabras muy comunes que pueden empezar con mayúscula (ej. 'Buenos')
            if nombre_potencial.lower() not in ['buenos', 'buenas', 'hola', 'soy', 'me']:
                 # Podríamos validar contra una lista de nombres comunes si fuera necesario
                 return nombre_potencial

    return None # No se encontró un patrón claro


# --- 4. Lógica Principal del Chatbot ---

def chatbot_financiero():
    print("\n=============================================")
    print("--- Iniciando Chatbot Financiero  ---")
    print("=============================================")
    print("\nChatbot:")
    for line in descripcion_general.split('\n'): print(line); time.sleep(0.05) 
    print("\n")
    time.sleep(0.5)

    # Inicialización del contexto y estado
    contexto = {campo: None for campo in CAMPOS_REQUERIDOS}
    contexto.update({'sector': None, 'valor_ingresos': None, 'valor_activos': None, 'moneda_ingresos': None, 'moneda_activos': None, 'moneda_cartera': None, 'moneda_deudas': None})
    confirmation_pending = None
    last_question_field = None
    campo_para_corregir = None
    user_name = None
    intent_history = [] # Para depuración o lógica futura

    # Iniciar conversación
    campo_siguiente = get_next_question(contexto)
    if campo_siguiente:
        saludo_inicial_template = random.choice(respuestas_saludo)
        saludo_inicial = format_greeting(saludo_inicial_template, None)
        pregunta_inicial = get_formatted_question(campo_siguiente) # Ya elige aleatoria
        print(f"Chatbot: {saludo_inicial.split('.')[0]}. {pregunta_inicial}") # Combina saludo y pregunta
        last_question_field = campo_siguiente
    else:
        print("Chatbot: Error inicial, no hay preguntas que hacer."); return

    # --- Bucle Principal ---
    while True:
        try:
            user_input = input("Tú: ").strip()
            if user_input.lower() in ['salir', 'exit', 'quit', 'terminar', 'adios', 'adiós']:
                print(f"Chatbot: {random.choice(respuestas_despedida)}")
                break
            print("-" * 40) # Separador visual

            # --- A. Manejo de Confirmación Pendiente ---
            if confirmation_pending:
                field_to_confirm = confirmation_pending['field']
                value_to_confirm = confirmation_pending['validated_value']
                currency = confirmation_pending.get('currency')
                sector_confirm = confirmation_pending.get('sector')

                # Tecnica Obligatoria: Lematización/POS usada aquí
                decision = check_confirmation(user_input)

                # Formatear valor para mostrar (después de check_confirmation)
                display_value = f"{value_to_confirm:,}" if isinstance(value_to_confirm, (int, float)) else str(value_to_confirm)
                if currency: display_value += f" {currency}"
                if field_to_confirm == 'area_categoria' and sector_confirm: display_value += f" (Sector: {sector_confirm})"

                # Combinar respuesta de acuse + siguiente pregunta
                response_parts = []

                if decision == 'yes':
                    ack_msg = random.choice(respuestas_acuse_recibo).format(campo_recibido=field_to_confirm.replace('_', ' '))
                    response_parts.append(ack_msg)
                    contexto[field_to_confirm] = value_to_confirm
                    if currency: contexto[f'moneda_{field_to_confirm.replace("valor_","")}'] = currency
                    if sector_confirm: contexto['sector'] = sector_confirm
                    if field_to_confirm in ['valor_ingresos', 'valor_activos']:
                        contexto['ingresos_o_activos'] = value_to_confirm
                        if field_to_confirm == 'valor_ingresos': contexto['valor_ingresos'] = value_to_confirm
                        if field_to_confirm == 'valor_activos': contexto['valor_activos'] = value_to_confirm

                    confirmation_pending = None
                    campo_siguiente = get_next_question(contexto)
                    if campo_siguiente:
                        response_parts.append(f"Continuemos. {get_formatted_question(campo_siguiente)}")
                        last_question_field = campo_siguiente
                    else:
                        response_parts.append("¡Excelente! Ya tengo toda la información necesaria. Analizando...")
                        print(f"Chatbot: {' '.join(response_parts)}")
                        break # Salir para predicción final

                elif decision == 'no':
                    correction_req = random.choice(respuestas_manejo_correccion).format(campo=field_to_confirm.replace('_', ' '))
                    response_parts.append(correction_req)
                    # Añadir sugerencia para nombre de empresa
                    if field_to_confirm == "nombre_empresa" and acronimos_sociedades:
                        acronimos_str = ", ".join([a['acronimo'] for a in acronimos_sociedades[:4]]) + ", etc."
                        response_parts.append(f"(Sugerencia: A veces ayuda usar una sigla de sociedad como {acronimos_str})")

                    confirmation_pending = None
                    campo_para_corregir = field_to_confirm
                    last_question_field = field_to_confirm

                else: # 'unclear'
                    response_parts.append("No entendí tu confirmación. Por favor, responde de forma más clara (ej: 'sí', 'correcto', 'no', 'falso').")
                    # Volver a preguntar la confirmación
                    confirm_template = random.choice(frases_confirmacion) # Elige una nueva forma de preguntar
                    response_parts.append(confirm_template.format(campo=field_to_confirm.replace('_', ' '), valor=display_value))


                print(f"Chatbot: {' '.join(response_parts)}") # Imprimir respuesta combinada
                print("-" * 40)
                continue

            # --- B. Manejo de Nueva Entrada ---
            intent = predict_intent(user_input)
            intent_history.append(intent) # Guardar historial
            print(f"DEBUG: Intención detectada: {intent}")

            response_parts = [] # Para construir la respuesta del bot

            if intent.startswith("Respuesta ") or campo_para_corregir:
                campo_esperado = campo_para_corregir if campo_para_corregir else last_question_field
                if not campo_esperado:
                     response_parts.append("Hmm, no estoy seguro de a qué pregunta responde eso. ¿Podrías ser más específico?")
                     print(f"Chatbot: {' '.join(response_parts)}")
                     print("-" * 40); continue

                print(f"DEBUG: Esperando respuesta para: {campo_esperado}")
                campo_para_corregir = None

                entities = predict_ner(user_input) # Llama a la función que ahora incluye demo Lema/POS
                print(f"DEBUG: Entidades NER encontradas: {entities}")

                dato_validado = None; texto_entidad = None; moneda_validada = None; sector_validado = None
                if campo_esperado == "valor_ingresos" or campo_esperado == "valor_activos":
                    buscar_campo = "ingresos_o_activos" # Usar el campo combinado
                else: buscar_campo = campo_esperado
                ner_labels_esperados = CAMPO_A_NER_LABEL.get(buscar_campo, [])
                if not isinstance(ner_labels_esperados, list): ner_labels_esperados = [ner_labels_esperados]

                entidades_encontradas = [entidad for entidad in entities if entidad['label'] in ner_labels_esperados]
                
                if entidades_encontradas:
                    texto_entidad = " ".join(entidad['text'] for entidad in entidades_encontradas).strip()
                    print(f"DEBUG: Texto de entidad principal: '{texto_entidad}' (Label: {entidades_encontradas[0]['label']})")

                    validation_successful = False
                    # (Lógica de validación interna )
                    if campo_esperado == "nombre_empresa": dato_validado = texto_entidad.strip(); validation_successful = bool(dato_validado)
                    elif campo_esperado == "numero_empleados": num_val, _ = parse_numero(texto_entidad); dato_validado = int(num_val) if num_val is not None and num_val >= 0 else None; validation_successful = (dato_validado is not None)
                    elif campo_esperado in ["valor_ingresos", "valor_activos", "valor_cartera", "valor_deudas"]:
                         num_val, mon_val = parse_numero(texto_entidad)
                         if num_val is not None and num_val >= 0: dato_validado = num_val; moneda_validada = mon_val; validation_successful = True
                         elif campo_esperado == 'valor_ingresos' and contexto['valor_activos'] is None: response_parts.append(f"No pude entender el valor de ingresos."); last_question_field = 'valor_activos'
                         elif campo_esperado == 'valor_activos' and contexto['valor_ingresos'] is None: response_parts.append(f"No pude entender el valor de activos."); last_question_field = 'valor_ingresos'
                    elif campo_esperado == "area_categoria":
                        # Busca la categoría más similar, SIN aplicar umbral aún
                        mejor_cat_match, mejor_sector_match, similitud = encontrar_mejor_categoria(texto_entidad)

                        if similitud >= SIMILARITY_THRESHOLD:
                            # --- Caso 1: Similitud ALTA ---
                            # Confiar en la categoría encontrada en la lista
                            print(f"DEBUG: Similitud ALTA ({similitud:.4f} >= {SIMILARITY_THRESHOLD}). Validando '{mejor_cat_match}'")
                            dato_validado = mejor_cat_match
                            sector_validado = mejor_sector_match
                            validation_successful = True

                        else:
                            # --- Caso 2: Similitud BAJA ---
                            # Preguntar al usuario si quiere usar SU TEXTO ORIGINAL
                            print(f"DEBUG: Similitud BAJA ({similitud:.4f} < {SIMILARITY_THRESHOLD}). Preguntando si mantener texto original.")
                            # Preparar para confirmar el TEXTO ORIGINAL del usuario
                            confirmation_pending = {
                                'field': campo_esperado, 
                                'validated_value': texto_entidad, 
                                'original_text': texto_entidad,
                                'currency': None, 
                                'sector': "No Especificado", 
                                'is_raw_text': True 
                            }
                            print(f"Chatbot: No encontré una categoría estándar que coincida bien con '{texto_entidad}'. ¿Quieres que usemos '{texto_entidad}' como la categoría de tu empresa? (Sí/No)")
                            validation_successful = False # Evita la confirmación estándar inmediata
                            print("-" * 40)
                            continue # Saltar al siguiente input del usuario

                    # --- Proceder a Confirmación si la validación fue exitosa (y no fue el caso de similitud baja) ---
                    if validation_successful:
                        # Este bloque ahora solo se ejecuta si similitud >= SIMILARITY_THRESHOLD
                        # O para otros tipos de campo (nombre, número, etc.)
                        confirmation_pending = {
                            'field': campo_esperado,
                            'validated_value': dato_validado,
                            'original_text': texto_entidad,
                            'currency': moneda_validada,
                            'sector': sector_validado, # Sector de la categoría encontrada o None
                            'is_raw_text': False # Indicar que NO es texto crudo
                        }
                        display_confirm = f"{dato_validado:,}" if isinstance(dato_validado, (int, float)) else str(dato_validado)
                        if moneda_validada: display_confirm += f" {moneda_validada}"
                        # Mostrar sector solo si se validó una categoría conocida
                        if campo_esperado == 'area_categoria' and sector_validado and not confirmation_pending.get('is_raw_text', False):
                             display_confirm += f" (Sector: {sector_validado})"

                        confirm_template = random.choice(frases_confirmacion)
                        response_parts.append(confirm_template.format(campo=campo_esperado.replace('_', ' '), valor=display_confirm) + " (Sí/No)")

                else: # No se encontraron entidades NER
                    response_not_found = f"No pude extraer la información de '{campo_esperado}' de tu respuesta.".replace('_', ' ')
                    response_parts.append(response_not_found)
                    if campo_esperado == "nombre_empresa" and acronimos_sociedades:
                        # Mostrar solo los primeros 4 acrónimos
                        acronimos_str = ", ".join([a['acronimo'] for a in acronimos_sociedades[:4]]) + ", etc."
                        response_parts.append(f"(Sugerencia: Me ayudas demasiado a entender el nombre si usas una sigla de sociedad como {acronimos_str}) al final del nombre de la empresa.")
                    response_parts.append(get_formatted_question(campo_esperado))
                    last_question_field = campo_esperado

            elif intent == "Saludo":
                # Intentar extraer nombre usando la función mejorada
                if contexto['nombre_empresa'] is None and not user_name: # Solo si aún no tenemos nombre
                     extracted_name = extract_name_from_greeting(user_input)
                     if extracted_name:
                          user_name = extracted_name
                          print(f"DEBUG: Nombre extraído del saludo: {user_name}")
                          
                saludo_template = random.choice(respuestas_saludo) 
                saludo_formateado = format_greeting(saludo_template, user_name)
                response_parts.append(saludo_formateado.split('¿')[0].strip()) # Saludo sin la pregunta original
                if last_question_field and contexto.get(last_question_field) is None:
                     response_parts.append(f"Continuemos. {get_formatted_question(last_question_field)}")

            elif intent == "Insulto":
                insulto_resp = random.choice(respuestas_insulto).split(':')[0].strip()
                response_parts.append(insulto_resp + ".")
                if last_question_field and contexto.get(last_question_field) is None:
                     response_parts.append(get_formatted_question(last_question_field))

            elif intent == "Pregunta Personal":
                personal_resp = random.choice(respuestas_pregunta_personal).split(':')[0].strip()
                response_parts.append(personal_resp + ".")
                if last_question_field and contexto.get(last_question_field) is None:
                     response_parts.append(get_formatted_question(last_question_field))

            elif intent == "Pregunta Económica":
                 term_found = next((term for term in terminos_economicos if term in user_input.lower()), None)
                 if term_found: response_parts.append(f"Claro, sobre '{term_found.capitalize()}': {terminos_economicos[term_found]}")
                 else: response_parts.append("Entiendo tu duda económica, pero no identifiqué el término. Prueba con 'Activos', 'Pasivos', etc.")
                 if last_question_field and contexto.get(last_question_field) is None: response_parts.append(f"Volviendo a lo nuestro... {get_formatted_question(last_question_field)}")

            elif intent == "Pregunta sobre Proceso":
                 field_topic_found = next((key for key, data in explicacion_proceso_campos.items() if any(kw in user_input.lower() for kw in key.split('_') + data.get('pregunta_asociada', [''])[0].lower().split() if len(kw)>3)), None)
                 if field_topic_found and field_topic_found in explicacion_proceso_campos: response_parts.append(f"Sobre '{field_topic_found.replace('_',' ')}': {explicacion_proceso_campos[field_topic_found]['explicacion']}")
                 else: response_parts.append(f"Entiendo que preguntas sobre el proceso. {descripcion_general.split('.')[0]}.")
                 if last_question_field and contexto.get(last_question_field) is None: response_parts.append(f"Continuemos... {get_formatted_question(last_question_field)}")

            else: # Intención desconocida
                response_parts.append("No estoy seguro de entender eso.")
                if last_question_field and contexto.get(last_question_field) is None:
                    response_parts.append(get_formatted_question(last_question_field))
                else:
                     campo_siguiente = get_next_question(contexto)
                     if campo_siguiente: response_parts.append(f"Quizás podríamos continuar con: {get_formatted_question(campo_siguiente)}"); last_question_field = campo_siguiente

            # Imprimir la respuesta combinada del bot
            print(f"Chatbot: {' '.join(response_parts)}")
            print("-" * 40)

        except Exception as e:
            print(f"\nChatbot: Ocurrió un error inesperado: {e}")
            import traceback; traceback.print_exc()
            # Intentar recuperar
            recovery_message = "Hubo un problema."
            if last_question_field and contexto.get(last_question_field) is None: recovery_message += f" Intentemos de nuevo con: {get_formatted_question(last_question_field)}"
            else:
                 campo_siguiente = get_next_question(contexto)
                 if campo_siguiente: recovery_message += f" Sigamos con: {get_formatted_question(campo_siguiente)}"; last_question_field = campo_siguiente
                 else: recovery_message += " No puedo continuar."; print(f"Chatbot: {recovery_message}"); break
            print(f"Chatbot: {recovery_message}")
            print("-" * 40)

    if get_next_question(contexto) is None:
        print("\n--- Realizando Análisis Final ---")
        lgbm_input_data = format_data_for_lgbm(contexto)
        if lgbm_input_data is not None:
            resultado_final_label = predict_lgbm(lgbm_input_data)
            print(f"DEBUG: Predicción LightGBM: {resultado_final_label}")
            respuestas_para_resultado = plantillas_resultado_final.get(resultado_final_label, ["No pude determinar una clasificación final con los datos proporcionados."])
            respuesta_final_elegida = random.choice(respuestas_para_resultado)
            print(f"\nChatbot: === Resultado del Análisis Financiero Preliminar ===")
            for char in respuesta_final_elegida: print(char, end='', flush=True); time.sleep(0.02) # Más rápido
            print("\n========================================================")
            print(f"Chatbot: {random.choice(respuestas_despedida)}")
        else:
            print("Chatbot: Hubo un problema al preparar los datos para el análisis final. No puedo darte una clasificación.")
            print(f"Chatbot: {random.choice(respuestas_despedida)}")
    else:
        print("\nChatbot: No se completó la recolección de datos.")
        print(f"Chatbot: {random.choice(respuestas_despedida)}")

if __name__ == "__main__":
    chatbot_financiero()