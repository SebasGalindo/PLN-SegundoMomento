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
# (Sin cambios en esta sección)
try:
    BASE_DIR = Path(__file__).resolve().parent # Directorio del script actual
    print(f"Directorio base: {BASE_DIR}")
    DATA_DIR = BASE_DIR.parent / "Data"
    if not DATA_DIR.exists(): # Si no está junto al script
        print(f"Advertencia: Directorio 'Data' no encontrado en {BASE_DIR}. Buscando en el directorio actual.")
        DATA_DIR = Path.cwd() / "Data"
        if not DATA_DIR.exists():
             raise FileNotFoundError("Directorio 'Data' no encontrado ni junto al script ni en el directorio actual.")
except NameError: # __file__ no está definido (ej. en un notebook interactivo)
    print("Advertencia: __file__ no definido. Buscando 'Data' en el directorio de trabajo actual.")
    DATA_DIR = Path.cwd() / "Data"
    if not DATA_DIR.exists():
         raise FileNotFoundError("Directorio 'Data' no encontrado en el directorio actual.")


INTENT_MODEL_DIR = DATA_DIR / "modelo_clasificador_frases"
NER_MODEL_DIR = DATA_DIR / "modelo_ner_frases"
LGBM_MODEL_PATH = DATA_DIR / "model_bundle_nivel_economico.joblib"
QUESTIONS_ANSWERS_PATH = DATA_DIR / "questions_and_answers_markdown.json" # Manteniendo tu ruta
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
    "area_categoria": ["CATEGORIA_EMPRESA", "SECTOR"], # Mantenido SECTOR por si el modelo NER lo usa
    "numero_empleados": ["NUM_EMPLEADOS"],
    "ingresos_o_activos": ["VALOR_GANANCIAS", "VALOR_ACTIVOS"],
    "valor_cartera": ["VALOR_CARTERA"],
    "valor_deudas": ["VALOR_DEUDAS"],
}
CAMPO_A_LISTA_PREGUNTAS = {}

LEMAS_POSITIVOS = {"sí", "si", "correcto", "verdadero", "afirmativo", "confirmar", "ok", "dale", "valer", "exacto", "así", "ser", "acuerdo", "proceder", "ajá", "efectivamente", "claro"}
LEMAS_NEGATIVOS = {"no", "incorrecto", "falso", "negativo", "error", "mal", "nunca", "jamás", "tampoco"}
PALABRAS_NEGACION_DIRECTA = {"no", "incorrecto", "falso", "negativo"}

# --- 2. Carga de Recursos (Se ejecuta una sola vez al importar) ---
print("--- Cargando Recursos del Chatbot (puede tardar un momento) ---")

# (Sin cambios en la carga de recursos: JSON, Modelos, spaCy, Tasas de cambio, etc.)
# 2.1 Cargar JSON y extraer datos
try:
    print(f"Intentando cargar JSON desde: {QUESTIONS_ANSWERS_PATH}")
    with open(QUESTIONS_ANSWERS_PATH, 'r', encoding='utf-8') as f:
        q_and_a_full = json.load(f)

    respuestas_saludo = q_and_a_full.get('respuestas_saludo', ["¡Hola! ¿Cuál es el nombre de tu empresa?"])
    respuestas_insulto = q_and_a_full.get('respuestas_insulto', ["Por favor, se respetuoso."])
    respuestas_pregunta_personal = q_and_a_full.get('respuestas_pregunta_personal', ["No puedo responder eso."])
    respuestas_acuse_recibo = q_and_a_full.get('respuestas_acuse_recibo', ["Ok, entendido {campo_recibido}."]) # Template añadido
    frases_confirmacion = q_and_a_full.get('frases_confirmacion', ["¿Es {valor} correcto para {campo}?"])
    respuestas_manejo_correccion = q_and_a_full.get('respuestas_manejo_correccion', ["Entendido. ¿Cuál sería el valor correcto para {campo}?"]) # Template añadido
    respuestas_despedida = q_and_a_full.get('respuestas_despedida', ["¡Adiós! Ha sido un placer ayudarte."])
    plantillas_resultado_final = q_and_a_full.get('plantillas_resultado_final', {})
    acronimos_sociedades = q_and_a_full.get('acronimos_sociedades', [])

    info_chatbot = q_and_a_full.get('informacion_chatbot', {})
    descripcion_general = info_chatbot.get('descripcion_general', "Soy un chatbot financiero diseñado para recopilar información de empresas y realizar un análisis preliminar.")
    terminos_economicos_list = info_chatbot.get('terminos_economicos', [])
    terminos_economicos = {item['termino'].lower(): item['definicion'] for item in terminos_economicos_list if 'termino' in item and 'definicion' in item}
    explicacion_proceso_campos = info_chatbot.get('explicacion_proceso_campos', {})

    for campo, data in explicacion_proceso_campos.items():
        if 'pregunta_asociada' in data and isinstance(data['pregunta_asociada'], list):
             CAMPO_A_LISTA_PREGUNTAS[campo] = data['pregunta_asociada']
        elif 'pregunta_asociada' in data:
             CAMPO_A_LISTA_PREGUNTAS[campo] = [data['pregunta_asociada']]
    print("JSON de preguntas y respuestas cargado.")
except FileNotFoundError:
    print(f"Error Crítico: No se encontró el archivo JSON en {QUESTIONS_ANSWERS_PATH}")
    raise # Opcional: detiene la app si el JSON no se encuentra
except (json.JSONDecodeError, KeyError, TypeError) as e:
    print(f"Error Crítico: El archivo JSON está mal formado o falta estructura clave: {e}")
    raise # Opcional

# 2.2 Cargar Modelo y Tokenizador de Clasificación de Intención
print(f"Cargando modelo de clasificación de intención desde: {MODELO_INTENCION}")
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
    raise # Opcional

# 2.3 Cargar Modelo y Tokenizador NER
print(f"Cargando modelo NER desde: {MODELO_NER}")
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
    raise # Opcional

# 2.4 Cargar Modelo LightGBM
print(f"Cargando modelo LightGBM desde: {LGBM_MODEL_PATH}")
try:
    lgbm_bundle = joblib.load(LGBM_MODEL_PATH)
    lgbm_model = lgbm_bundle['model']
    lgbm_label_encoder = lgbm_bundle['label_encoder']
    lgbm_features = lgbm_bundle['features']
    print("Modelo LightGBM cargado.")
    print(f"  -> Features esperadas por LightGBM: {lgbm_features}")
except FileNotFoundError:
     print(f"Error Crítico: No se encontró el archivo del modelo LightGBM en {LGBM_MODEL_PATH}")
     raise # Opcional
except Exception as e:
     print(f"Error Crítico cargando el modelo LightGBM: {e}")
     raise # Opcional

# 2.5 Cargar Modelo spaCy y Categorías con Embeddings
print(f"Cargando modelo spaCy '{MODELO_SPACY}'...")
try:
    nlp = spacy.load(MODELO_SPACY)
    print("Modelo spaCy cargado.")

    print(f"Cargando categorías desde: {CATEGORIES_PATH}")
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
    print(f"{len(categorias_empresa)} categorías vectorizadas.")
    if not categorias_empresa:
        print("Error Crítico: No se pudieron vectorizar categorías.")
        raise ValueError("No se pudieron cargar/vectorizar categorías de empresa.") # Opcional

except OSError:
     print(f"Error Crítico: Modelo spaCy '{MODELO_SPACY}' no encontrado. Descárgalo con:")
     print(f"python -m spacy download {MODELO_SPACY}")
     raise # Opcional
except FileNotFoundError:
     print(f"Error Crítico: No se encontró el archivo de categorías en {CATEGORIES_PATH}")
     raise # Opcional
except Exception as e:
     print(f"Error Crítico cargando spaCy o categorías: {e}")
     raise # Opcional

# 2.6 Configurar Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
intent_model.to(device)
intent_model.eval()
ner_model.to(device)
ner_model.eval()
print(f"\nModelos Transformers listos en dispositivo: {device}")

# 2.7 Obtener Tasas de Cambio (Yahoo Finance)
print(f"\nObteniendo tasas de cambio desde Yahoo Finance...")
ticker_usd_cop = 'USDCOP=X'
ticker_eur_cop = 'EURCOP=X'
tasas = {'USD_COP': 4000.0, 'EUR_COP': 4500.0} # Valores por defecto
try:
    print(f"Buscando ticker: {ticker_usd_cop}...")
    ticker_obj_usd = yf.Ticker(ticker_usd_cop)
    info_usd = ticker_obj_usd.info
    if 'regularMarketPrice' in info_usd and info_usd['regularMarketPrice'] is not None:
         tasas['USD_COP'] = info_usd['regularMarketPrice']
         print(f"  Tasa USD/COP encontrada: {tasas['USD_COP']:,.2f}")
    else: print(f"  No se pudo obtener tasa USD/COP reciente, usando defecto: {tasas['USD_COP']}")

    print(f"Buscando ticker: {ticker_eur_cop}...")
    ticker_obj_eur = yf.Ticker(ticker_eur_cop)
    info_eur = ticker_obj_eur.info
    if 'regularMarketPrice' in info_eur and info_eur['regularMarketPrice'] is not None:
         tasas['EUR_COP'] = info_eur['regularMarketPrice']
         print(f"  Tasa EUR/COP encontrada: {tasas['EUR_COP']:,.2f}")
    else: print(f"  No se pudo obtener tasa EUR/COP reciente, usando defecto: {tasas['EUR_COP']}")

except Exception as e:
    print(f"Error obteniendo datos de Yahoo Finance (se usarán tasas por defecto): {e}")

print("\n--- Todos los recursos cargados exitosamente ---")

# --- 3. Funciones Auxiliares ---
# (Todas las funciones auxiliares: predict_intent, predict_ner, parse_numero,
# encontrar_mejor_categoria, get_next_question_key, get_formatted_question,
# format_greeting, check_confirmation, convert_to_COP, format_data_for_lgbm,
# predict_lgbm, extract_name_from_greeting permanecen EXACTAMENTE IGUAL que en
# tu versión original de chatbot_logic.py)

def predict_intent(text):
    # 
    inputs = intent_tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=MAX_LEN_BERT)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = intent_model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    return intent_id2label.get(predicted_class_id, "Intencion Desconocida")

def predict_ner(text):
    # 
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
    'cien': 100, 'ciento': 100, 'cientos': 100,
    'doscientos': 200, 'trescientos': 300, 'trecientos': 300,
    'cuatrocientos': 400, 'quinientos': 500, 'seiscientos': 600,
    'setecientos': 700, 'ochocientos': 800, 'novecientos': 900
}
conectores_spacy = {'y', 'de'}

def parse_numero(texto):
    # 
    if not texto or not isinstance(texto, str): return None, None
    texto_original = texto; #print(f"DEBUG parse_numero: Recibido '{texto_original}'")
    texto_limpio = texto.lower().strip(); texto_limpio = re.sub(r'[$\']', '', texto_limpio); texto_limpio = re.sub(r'\s+', ' ', texto_limpio)
    monedas_map = {'pesos': 'COP', 'peso': 'COP', 'dolar': 'USD', 'dólar': 'USD', 'dolares': 'USD', 'dólares': 'USD', 'eur': 'EUR', 'euro': 'EUR', 'euros': 'EUR', 'cop': 'COP', 'usd': 'USD'}
    moneda = DEFAULT_CURRENCY; texto_sin_moneda = texto_limpio; moneda_detectada_str = None
    for palabra, codigo in monedas_map.items():
        pattern = r'\b' + re.escape(palabra) + r'\b'; match = re.search(pattern, texto_limpio)
        if match: moneda = codigo; moneda_detectada_str = match.group(0); texto_sin_moneda = re.sub(pattern, '', texto_sin_moneda).strip(); break #print(f"DEBUG parse_numero: Moneda detectada: {moneda}"); break
    relleno = ['unos', 'aproximadamente', 'cerca de', 'alrededor de', 'como', 'un total de', 'más o menos', 'de'];
    for palabra_relleno in relleno: texto_sin_moneda = re.sub(r'\b' + re.escape(palabra_relleno) + r'\b', '', texto_sin_moneda).strip()
    texto_sin_moneda = re.sub(r'\s+', ' ', texto_sin_moneda).strip(); #print(f"DEBUG parse_numero: Texto para procesar: '{texto_sin_moneda}'")
    doc = nlp(texto_sin_moneda); total_valor = 0.0; valor_segmento = 0.0; segmento_con_digito = False
    for token in doc:
        #print(f"  Procesando Token: '{token.text}' (Lema: {token.lemma_}, like_num: {token.like_num})")
        if token.like_num:
            try:
                num_text = token.text.replace(',', '.').replace(' ', '');
                if num_text.count('.') > 1: num_text = num_text.replace('.', '', num_text.count('.') - 1) # Mantener el último punto como decimal
                num_digit = float(num_text); valor_segmento = num_digit; segmento_con_digito = True; #print(f"    -> Detectado como dígito: {num_digit}. valor_segmento={valor_segmento}")
            except ValueError: segmento_con_digito = False #print(f"    -> 'like_num' pero no se pudo convertir a float: '{token.text}'")
        if token.lemma_ in numeros_texto_spacy and not segmento_con_digito:
             if token.lemma_ in ['ciento', 'cien', 'cientos']:
                 if valor_segmento > 0: valor_segmento *= 100
                 else: valor_segmento += 100
             else: 
                 valor_segmento += numeros_texto_spacy[token.lemma_] #print(f"    -> Detectado como palabra numérica: {numeros_texto_spacy[token.lemma_]}. valor_segmento={valor_segmento}")
        if token.lemma_ in multiplicadores_spacy:
            if valor_segmento == 0: valor_segmento = 1; #print(f"    -> Multiplicador sin número previo, usando 1.")
            valor_aplicado = valor_segmento * multiplicadores_spacy[token.lemma_]; total_valor += valor_aplicado; #print(f"    -> Aplicando multiplicador '{token.lemma_}'. Sumando {valor_aplicado}. total_valor={total_valor}")
            valor_segmento = 0; segmento_con_digito = False
        if token.lemma_ not in conectores_spacy and not (token.lemma_ in numeros_texto_spacy or token.lemma_ in multiplicadores_spacy or token.like_num):
             #print(f"    -> Palabra '{token.text}' (lema: {token.lemma_}) ignorada.")
             if valor_segmento > 0: total_valor += valor_segmento; valor_segmento = 0; segmento_con_digito = False #print(f"    -> Palabra desconocida encontrada, sumando segmento pendiente: {valor_segmento}")
    if valor_segmento > 0: total_valor += valor_segmento; #print(f"DEBUG parse_numero: Sumando segmento final remanente: {valor_segmento}. total_valor={total_valor}")
    if total_valor > 0: #print(f"DEBUG parse_numero: Valor final extraído: {total_valor}");
        return total_valor, moneda
    else:
        try:
            num_str_directo = texto_sin_moneda.replace('.', '').replace(',', '.').replace(' ', '');
            if num_str_directo: valor = float(num_str_directo); return valor, moneda #print(f"DEBUG parse_numero: Fallback final funcionó: {valor}"); return valor, moneda
        except ValueError: pass #print(f"DEBUG parse_numero: No se pudo extraer un valor numérico de '{texto_original}'.")
        return None, None

def encontrar_mejor_categoria(texto_categoria):
    # 
    #print(f"\n--- Buscando similitud para: '{texto_categoria}' (Embeddings spaCy) ---")
    doc_usuario = nlp(texto_categoria.lower())
    if not doc_usuario.has_vector or not doc_usuario.vector_norm: return None, None, 0.0
    vector_usuario_norm = doc_usuario.vector / doc_usuario.vector_norm
    mejor_cat_encontrada = None; mejor_sector_encontrado = None; max_sim = -1.0
    for cat_original, data in categorias_empresa.items():
        similitud = np.dot(vector_usuario_norm, data['vector'])
        #print(f"  Comparando con '{cat_original}': Similitud = {similitud:.4f}")
        if similitud > max_sim: max_sim = similitud; mejor_cat_encontrada = cat_original; mejor_sector_encontrado = data['sector']
    #print(f"DEBUG: Mejor coincidencia preliminar: '{mejor_cat_encontrada}' (Sector: {mejor_sector_encontrado}, Sim: {max_sim:.4f})")
    return mejor_cat_encontrada, mejor_sector_encontrado, max_sim

def get_next_question_key(contexto):
    """Devuelve la *clave* del siguiente campo requerido, o None si todo está completo."""
    # 
    for campo in CAMPOS_REQUERIDOS:
        if campo == 'ingresos_o_activos':
             if contexto.get('valor_ingresos') is None and contexto.get('valor_activos') is None:
                 return 'valor_ingresos'
        elif contexto.get(campo) is None:
            return campo
    return None

def get_formatted_question(campo_key):
    # 
    lista_preguntas = CAMPO_A_LISTA_PREGUNTAS.get(campo_key)
    if lista_preguntas and isinstance(lista_preguntas, list):
        pregunta = random.choice(lista_preguntas)
    else:
        pregunta = f"¿Podrías darme información sobre {campo_key.replace('_', ' ')}?"
        print(f"ADVERTENCIA: No se encontró lista de preguntas para '{campo_key}', usando fallback.")
    if campo_key in explicacion_proceso_campos:
         ejemplo = explicacion_proceso_campos[campo_key].get('ejemplo_respuesta')
         if ejemplo: pregunta += f" (Ej: {ejemplo})"
    return pregunta

def format_greeting(template, name=None):
    # 
    if name: return template.replace("nombre", name)
    else: return re.sub(r'\s?nombre[!?]?', '', template).strip()

def check_confirmation(text):
    # 
    #print("\n--- Analizando Confirmación/Negación (Lema/POS) ---")
    doc = nlp(text.lower()); has_negation = False; has_affirmation = False
    for token in doc:
        #print(f"  Token: {token.text}, Lema: {token.lemma_}, POS: {token.pos_}")
        if token.lemma_ in LEMAS_NEGATIVOS or token.text in PALABRAS_NEGACION_DIRECTA:
            is_double_negation = False
            if token.i > 0 and doc[token.i-1].lemma_ == "ser":
                 if token.i + 1 < len(doc) and doc[token.i+1].lemma_ in LEMAS_NEGATIVOS: is_double_negation = True
            if not is_double_negation: has_negation = True; break #print("  -> Negación detectada.") break
        if token.lemma_ in LEMAS_POSITIVOS: has_affirmation = True #print("  -> Afirmación detectada.")
    #print("-------------------------------------------------")
    if has_negation: return 'no'
    elif has_affirmation: return 'yes'
    else: return 'unclear'

def convert_to_COP(valor, moneda):
    # 
    if moneda == 'COP' or valor is None:
        return valor
    tasa = tasas.get(f'{moneda}_COP')
    if tasa:
        valor_convertido = valor * tasa
        #print(f"DEBUG: Convertido {valor} {moneda} a {valor_convertido:.2f} COP usando tasa {tasa:.2f}.")
        return valor_convertido
    else:
        print(f"ADVERTENCIA: No se encontró tasa de cambio para {moneda}, se usará el valor original.")
        return valor

def format_data_for_lgbm(contexto):
    # 
    try:
        valor_activo_o_ingreso = 0
        moneda_ref = DEFAULT_CURRENCY
        if contexto.get('valor_activos') is not None:
             valor_activo_o_ingreso = contexto['valor_activos']
             moneda_ref = contexto.get('moneda_activos', DEFAULT_CURRENCY)
        elif contexto.get('valor_ingresos') is not None:
             valor_activo_o_ingreso = contexto['valor_ingresos']
             moneda_ref = contexto.get('moneda_ingresos', DEFAULT_CURRENCY)

        data = {
            'Sector': [contexto.get('sector', 'Desconocido')],
            'Area': [contexto.get('area_categoria', 'Desconocida')],
            'Numero Empleados': [contexto.get('numero_empleados', 0)],
            'Activos (COP)': [convert_to_COP(valor_activo_o_ingreso, moneda_ref)],
            'Cartera (COP)': [convert_to_COP(contexto.get('valor_cartera', 0), contexto.get('moneda_cartera', DEFAULT_CURRENCY))],
            'Deudas (COP)': [convert_to_COP(contexto.get('valor_deudas', 0), contexto.get('moneda_deudas', DEFAULT_CURRENCY))],
            }
        df_row = pd.DataFrame.from_dict(data); #print(f"DataFrame inicial para LGBM:\n{df_row}")
        missing_cols = [col for col in lgbm_features if col not in df_row.columns]
        if missing_cols:
             print(f"  ERROR: Faltan columnas esperadas por LGBM: {missing_cols}. Completando con valores por defecto.");
             for col in missing_cols: df_row[col] = 0 # O manejar de otra forma

        for col in lgbm_features:
             if col not in df_row.columns: continue # Ya manejado arriba
             if df_row[col].dtype.name in ['object', 'category']: df_row[col] = df_row[col].astype('category')
             else: df_row[col] = pd.to_numeric(df_row[col], errors='coerce').fillna(0)

        df_row = df_row[lgbm_features]; #print(f"DataFrame final formateado para LGBM:\n{df_row}"); #print(f"Tipos de datos finales: \n{df_row.dtypes}");
        return df_row
    except Exception as e: print(f"Error formateando datos para LightGBM: {e}"); return None

def predict_lgbm(data_row):
    # 
    try:
        for col in data_row.select_dtypes(include='category').columns:
            pass
        prediction_encoded = lgbm_model.predict(data_row)[0]
        prediction_label = lgbm_label_encoder.inverse_transform([prediction_encoded])[0]
        return prediction_label
    except ValueError as ve:
        print(f"Error de Valor durante predicción LightGBM: {ve}")
        print("  Posible causa: Categoría no vista durante el entrenamiento.")
        return "Error: Categoría Desconocida"
    except Exception as e: print(f"Error durante la predicción LightGBM: {e}"); return "Error en Predicción"



PALABRAS_COMUNES_A_EXCLUIR = {
    # Artículos
    'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'lo',
    # Preposiciones
    'a', 'ante', 'bajo', 'cabe', 'con', 'contra', 'de', 'desde', 'durante',
    'en', 'entre', 'hacia', 'hasta', 'mediante', 'para', 'por', 'segun',
    'sin', 'so', 'sobre', 'tras', 'versus', 'via',
    # Conjunciones
    'y', 'e', 'ni', 'o', 'u', 'pero', 'mas', 'sino', 'aunque', 'porque',
    'pues', 'como', 'si', 'que', 'cuando',
    # Adverbios comunes
    'aqui', 'aca', 'alli', 'ahi', 'alla', 'antes', 'despues', 'luego', 'pronto',
    'tarde', 'temprano', 'ayer', 'hoy', 'mañana', 'siempre', 'nunca', 'jamas',
    'ya', 'mientras', 'todavia', 'aun', 'recien', 'pronto', 'cerca', 'lejos',
    'encima', 'debajo', 'delante', 'detras', 'enfrente', 'alrededor', 'asi',
    'bien', 'mal', 'peor', 'mejor', 'despacio', 'deprisa', 'muy', 'mucho',
    'poco', 'bastante', 'mas', 'menos', 'algo', 'nada', 'solo', 'tambien',
    'tampoco', 'si', 'no', 'quizas', 'acaso', 'talvez', 'ademas', 'incluso',
    'excepto', 'salvo', 'inclusive',
    # Pronombres
    'yo', 'tu', 'el', 'ella', 'usted', 'nosotros', 'nosotras', 'vosotros',
    'vosotras', 'ellos', 'ellas', 'ustedes', 'me', 'te', 'se', 'nos', 'os',
    'le', 'les', 'mi', 'mis', 'tu', 'tus', 'su', 'sus', 'mio', 'mios', 'mia',
    'mias', 'tuyo', 'tuya', 'tuyos', 'tuyas', 'suyo', 'suya', 'suyos', 'suyas',
    'nuestro', 'nuestra', 'nuestros', 'nuestras', 'vuestro', 'vuestra',
    'vuestros', 'vuestras', 'este', 'esta', 'estos', 'estas', 'ese', 'esa',
    'esos', 'esas', 'aquel', 'aquella', 'aquellos', 'aquellas', 'esto', 'eso',
    'aquello', 'alguien', 'nadie', 'algo', 'nada', 'cualquier', 'cualquiera',
    'quienquiera', 'quien', 'cual', 'cuyo',
    # Interjecciones y Saludos (ya cubiertos en patrones, pero por si acaso)
    'hola', 'hey', 'epa', 'oye', 'buenos', 'buenas', 'adios', 'chao', 'hasta',
    'anda', 'vaya', 'hombre', 'mujer', 'caramba', 'ay', 'oh', 'uf', 'eh',
    # Verbos comunes (infinitivos y algunas formas conjugadas)
    'soy', 'eres', 'es', 'somos', 'sois', 'son', 'estoy', 'estas', 'esta',
    'estamos', 'estais', 'estan', 'he', 'has', 'ha', 'hemos', 'habeis', 'han',
    'ser', 'estar', 'haber', 'tener', 'hacer', 'decir', 'ir', 'ver', 'dar',
    'saber', 'querer', 'llegar', 'pasar', 'deber', 'poner', 'parecer', 'quedar',
    'creer', 'hablar', 'llevar', 'dejar', 'seguir', 'encontrar', 'llamar',
    'venir', 'pensar', 'salir', 'volver', 'tomar', 'conocer', 'vivir', 'sentir',
    'tratar', 'mirar', 'contar', 'empezar', 'esperar', 'buscar', 'entrar',
    'trabajar', 'escribir', 'perder', 'producir', 'ocurrir', 'recibir',
    'recordar', 'terminar', 'permitir', 'aparecer', 'conseguir', 'comenzar',
    'servir', 'sacar', 'necesitar', 'mantener', 'resultar', 'leer', 'caer',
    'cambiar', 'presentar', 'crear', 'abrir', 'considerar', 'oir', 'acabar',
    'convertir', 'ganar', 'formar', 'traer', 'partir', 'morir', 'aceptar',
    'realizar', 'suponer', 'comprender', 'lograr', 'explicar', 'preguntar',
    'tocar', 'reconocer', 'estudiar', 'alcanzar', 'nacer', 'dirigir', 'correr',
    'utilizar', 'pagar', 'ayudar', 'gustar', 'jugar', 'escuchar', 'levantar',
    'intentar', 'usar',
    # Otros comunes que pueden dar problemas
    'favor', 'gracias', 'porfa', 'dia', 'dias', 'tarde', 'tardes', 'noche',
    'noches', 'tal', 'como', 'va', 'numero', 'identificacion', 'nombre',
    'llamo', 'apellido', 'aqui', 'habla', 'saluda', 'saludo', 'dice',
    'mensaje', 'texto', 'correo', 'electronico'
}

def extract_name_from_greeting(text):
    """
    Extrae un nombre de una cadena de texto que puede contener saludos o introducciones.

    Intenta identificar nombres basándose en patrones comunes en español,
    evitando confundir palabras comunes (verbos, saludos, preposiciones, etc.) con nombres.

    Args:
        text (str): El texto de entrada.

    Returns:
        str or None: El nombre extraído (con su capitalización original si es posible),
                     o None si no se encuentra un nombre probable.
    """
    if not text:
        return None

    patterns = [
        # 1. "me llamo/mi nombre es/soy [Nombre Apellido]"
        r'(?:me\s+llamo|mi\s+nombre\s+es|soy)\s+([A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+(?:\s+[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+)*)\b',
        # 2. "[Saludo], [soy/me llamo/etc] [Nombre Apellido]" 
        r'(?:buenos\s+d[íi]as|buenas\s+tardes|buenas\s+noches|hola|hey|qu[ée]\s+tal|buenas|saludos)[,]?\s*(?:soy|me\s+llamo|habla|le\s+saluda|por\s+aqu[íi])?\s*([A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+(?:\s+[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+)*)\b',
        # 3. "[Nombre Apellido] [habla/le saluda/por aquí]"
        r'\b([A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+(?:\s+[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+)*)\s+(?:habla|le\s+saluda|por\s+aqu[íi])',
        # 4. Patrón más general al inicio (UNA o DOS palabras capitalizadas)
        # Se usa (?:\s+[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+)? para hacerlo opcional el segundo nombre/apellido
        r'^\s*([A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+(?:\s+[A-ZÁÉÍÓÚÑÜ][a-záéíóúñü]+)?)\b'
    ]

    for i, pattern in enumerate(patterns):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            nombre_potencial = match.groups()[-1]

            if nombre_potencial: 
                nombre_limpio = nombre_potencial.strip()

                # Validar contra la lista de exclusión
                # Comprobar la primera palabra y el nombre completo (en minúsculas)
                primera_palabra = nombre_limpio.split()[0].lower()
                nombre_completo_lower = nombre_limpio.lower()

                # Si ni la primera palabra ni el nombre completo 
                if primera_palabra not in PALABRAS_COMUNES_A_EXCLUIR and \
                   (len(nombre_limpio.split()) > 1 or nombre_completo_lower not in PALABRAS_COMUNES_A_EXCLUIR):
                    return nombre_limpio

    # Si ningún patrón funcionó o todos los candidatos fueron excluidos
    return None

# --- 4. Funciones Principales para Streamlit ---

def get_initial_state():
    """Devuelve el diccionario de estado inicial para una nueva conversación."""
    initial_context = {campo: None for campo in CAMPOS_REQUERIDOS}
    initial_context.update({
        'sector': None, 'valor_ingresos': None, 'valor_activos': None,
        'moneda_ingresos': None, 'moneda_activos': None,
        'moneda_cartera': None, 'moneda_deudas': None
    })
    return {
        "contexto": initial_context,
        "confirmation_pending": None,
        "last_question_field": None,
        "campo_para_corregir": None,
        "user_name": None,
        "intent_history": [],
        "conversation_complete": False
    }

def get_initial_message(state):
    """Genera el mensaje de bienvenida y la primera pregunta."""
    campo_siguiente = get_next_question_key(state['contexto'])
    if campo_siguiente:
        saludo_inicial_template = random.choice(respuestas_saludo)
        saludo_inicial = format_greeting(saludo_inicial_template, state.get('user_name'))
        pregunta_inicial = get_formatted_question(campo_siguiente)
        state['last_question_field'] = campo_siguiente
        saludo_base = saludo_inicial.split('¿')[0].strip('.?! ')
        return f"{descripcion_general} {saludo_base}. {pregunta_inicial}", state
    else:
        state['conversation_complete'] = True
        return "Parece que no necesito ninguna información. ¿Hay algo más en lo que pueda ayudarte?", state


def process_user_input(user_input, current_state):
    """
    Procesa una entrada del usuario, actualiza el estado y devuelve la respuesta del chatbot.
    Args: user_input (str), current_state (dict)
    Returns: tuple: (str, dict) La respuesta del chatbot y el estado actualizado.
    """

    state = current_state.copy() 
    contexto = state['contexto']
    confirmation_pending = state['confirmation_pending']
    last_question_field = state['last_question_field']
    campo_para_corregir = state['campo_para_corregir']
    user_name = state['user_name']
    response_parts = []


    if confirmation_pending:
        field_to_confirm = confirmation_pending['field']
        value_to_confirm = confirmation_pending['validated_value']
        currency = confirmation_pending.get('currency')
        sector_confirm = confirmation_pending.get('sector')
        is_raw_text = confirmation_pending.get('is_raw_text', False)

        decision = check_confirmation(user_input)

        display_value = f"{value_to_confirm:,}" if isinstance(value_to_confirm, (int, float)) else str(value_to_confirm)
        if currency: display_value += f" {currency}"
        if field_to_confirm == 'area_categoria':
             if sector_confirm and not is_raw_text: display_value += f" (Sector: {sector_confirm})"

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

            state['confirmation_pending'] = None
            campo_siguiente_key = get_next_question_key(contexto)
            if campo_siguiente_key:
                next_question_text = get_formatted_question(campo_siguiente_key)
                response_parts.append(f"Continuemos. {next_question_text}")
                state['last_question_field'] = campo_siguiente_key
            else:
                state['conversation_complete'] = True
                response_parts.append("¡Excelente! Ya tengo toda la información necesaria. Permíteme analizarla...")

        elif decision == 'no':
            correction_req = random.choice(respuestas_manejo_correccion).format(campo=field_to_confirm.replace('_', ' '))
            response_parts.append(correction_req)
            if field_to_confirm == "nombre_empresa" and acronimos_sociedades:
                acronimos_str = ", ".join([a['acronimo'] for a in acronimos_sociedades[:4]]) + ", etc."
                response_parts.append(f"(Sugerencia: Ayuda usar una sigla como {acronimos_str})")

            state['confirmation_pending'] = None
            state['campo_para_corregir'] = field_to_confirm
            state['last_question_field'] = field_to_confirm

        else: 
            response_parts.append("No entendí tu confirmación. Por favor, responde 'sí' o 'no'.")
            confirm_template = random.choice(frases_confirmacion)
            response_parts.append(confirm_template.format(campo=field_to_confirm.replace('_', ' '), valor=display_value) + " (Sí/No)")

        return " ".join(response_parts), state


    intent = predict_intent(user_input)
    state['intent_history'].append(intent)

    if user_input.lower() in ['salir', 'exit', 'quit', 'terminar', 'adios', 'adiós']:
        state['conversation_complete'] = True 
        return random.choice(respuestas_despedida), state

    
    if intent.startswith("Respuesta ") or state['campo_para_corregir']:
        campo_esperado = state['campo_para_corregir'] if state['campo_para_corregir'] else state['last_question_field']
        if not campo_esperado:
             response_parts.append("Hmm, no estoy seguro de a qué pregunta responde eso.")
             campo_siguiente_key = get_next_question_key(contexto)
             if campo_siguiente_key:
                  response_parts.append(f"Quizás podríamos continuar con: {get_formatted_question(campo_siguiente_key)}")
                  state['last_question_field'] = campo_siguiente_key
             return " ".join(response_parts), state

        state['campo_para_corregir'] = None 

        entities = predict_ner(user_input)
        dato_validado = None; texto_entidad = None; moneda_validada = None; sector_validado = None
        validation_successful = False

        buscar_campo_ner_key = campo_esperado
        if campo_esperado == "valor_ingresos" or campo_esperado == "valor_activos":
            buscar_campo_ner_key = "ingresos_o_activos"
        elif campo_esperado not in CAMPO_A_NER_LABEL:
             buscar_campo_ner_key = None

        ner_labels_esperados = []
        if buscar_campo_ner_key:
             ner_labels_esperados = CAMPO_A_NER_LABEL.get(buscar_campo_ner_key, [])
             if not isinstance(ner_labels_esperados, list): ner_labels_esperados = [ner_labels_esperados]

        entidades_relevantes = []
        if ner_labels_esperados:
             entidades_relevantes = [entidad for entidad in entities if entidad['label'] in ner_labels_esperados]
        else:
             if entities:
                  entidades_relevantes = [entities[0]]

        if entidades_relevantes:
            texto_entidad = " ".join(entidad['text'] for entidad in entidades_relevantes).strip()

            if campo_esperado == "nombre_empresa":
                dato_validado = texto_entidad.strip()
                validation_successful = bool(dato_validado)

            elif campo_esperado == "numero_empleados":
                num_val, _ = parse_numero(texto_entidad)
                if num_val is not None and num_val >= 0:
                     dato_validado = int(round(num_val))
                     validation_successful = True
                else: 
                    num_val_full, _ = parse_numero(user_input)
                    if num_val_full is not None and num_val_full >= 0:
                        dato_validado = int(round(num_val_full))
                        validation_successful = True


            elif campo_esperado in ["valor_ingresos", "valor_activos", "valor_cartera", "valor_deudas"]:
                 num_val, mon_val = parse_numero(texto_entidad)
                 if num_val is not None and num_val >= 0:
                      dato_validado = num_val
                      moneda_validada = mon_val if mon_val else DEFAULT_CURRENCY
                      validation_successful = True
                 else: 
                     num_val_full, mon_val_full = parse_numero(user_input)
                     if num_val_full is not None and num_val_full >= 0:
                          dato_validado = num_val_full
                          moneda_validada = mon_val_full if mon_val_full else DEFAULT_CURRENCY
                          validation_successful = True

            elif campo_esperado == "area_categoria":
                texto_para_similitud = texto_entidad if texto_entidad else user_input
                mejor_cat_match, mejor_sector_match, similitud = encontrar_mejor_categoria(texto_para_similitud)

                if mejor_cat_match and similitud >= SIMILARITY_THRESHOLD:
                    dato_validado = mejor_cat_match
                    sector_validado = mejor_sector_match
                    validation_successful = True
                else:
                    state['confirmation_pending'] = {
                        'field': campo_esperado,
                        'validated_value': texto_para_similitud.strip(),
                        'original_text': texto_para_similitud.strip(),
                        'currency': None,
                        'sector': "No Especificado",
                        'is_raw_text': True
                    }
                    response_parts.append(f"No encontré una categoría estándar que coincida bien con '{texto_para_similitud}'. ¿Quieres que usemos '{texto_para_similitud.strip()}' como la categoría? (Sí/No)")
                    return " ".join(response_parts), state

            # --- Fin Validación ---

            if validation_successful:
                state['confirmation_pending'] = {
                    'field': campo_esperado,
                    'validated_value': dato_validado,
                    'original_text': texto_entidad if texto_entidad else user_input,
                    'currency': moneda_validada,
                    'sector': sector_validado,
                    'is_raw_text': False
                }
                display_confirm = f"{dato_validado:,}" if isinstance(dato_validado, (int, float)) else str(dato_validado)
                if moneda_validada: display_confirm += f" {moneda_validada}"
                if campo_esperado == 'area_categoria' and sector_validado: display_confirm += f" (Sector: {sector_validado})"

                confirm_template = random.choice(frases_confirmacion)
                response_parts.append(confirm_template.format(campo=campo_esperado.replace('_', ' '), valor=display_confirm) + " (Sí/No)")
            elif not state.get('confirmation_pending'):
                 response_parts.append(f"No pude extraer información válida para '{campo_esperado.replace('_', ' ')}' de tu respuesta.")
                 if campo_esperado == "nombre_empresa" and acronimos_sociedades:
                     acronimos_str = ", ".join([a['acronimo'] for a in acronimos_sociedades[:4]]) + ", etc."
                     response_parts.append(f"(Sugerencia: Ayuda usar una sigla como {acronimos_str})")
                 response_parts.append(f"Intentemos de nuevo: {get_formatted_question(campo_esperado)}")
                 state['last_question_field'] = campo_esperado

        else: 
            response_parts.append(f"No pude identificar la información clave para '{campo_esperado.replace('_', ' ')}' en tu mensaje.")
            fallback_parsed = False
            if campo_esperado == "numero_empleados":
                num_val_full, _ = parse_numero(user_input)
                if num_val_full is not None and num_val_full >= 0:
                    dato_validado = int(round(num_val_full)); validation_successful = True; fallback_parsed = True
                    
            elif campo_esperado in ["valor_ingresos", "valor_activos", "valor_cartera", "valor_deudas"]:
                 num_val_full, mon_val_full = parse_numero(user_input)
                 if num_val_full is not None and num_val_full >= 0:
                     dato_validado = num_val_full; moneda_validada = mon_val_full if mon_val_full else DEFAULT_CURRENCY
                     validation_successful = True; fallback_parsed = True
                    
            elif campo_esperado == "area_categoria":
                pass

            if fallback_parsed and validation_successful:
                 state['confirmation_pending'] = {
                    'field': campo_esperado, 'validated_value': dato_validado,
                    'original_text': user_input, 'currency': moneda_validada,
                    'sector': None, 'is_raw_text': False }
                 display_confirm = f"{dato_validado:,}" if isinstance(dato_validado, (int, float)) else str(dato_validado)
                 if moneda_validada: display_confirm += f" {moneda_validada}"
                 confirm_template = random.choice(frases_confirmacion)
                 response_parts = [confirm_template.format(campo=campo_esperado.replace('_', ' '), valor=display_confirm) + " (Sí/No)"]
            else:
                if campo_esperado == "nombre_empresa" and acronimos_sociedades:
                    acronimos_str = ", ".join([a['acronimo'] for a in acronimos_sociedades[:4]]) + ", etc."
                    response_parts.append(f"(Sugerencia: Para nombres de empresa, ayuda usar una sigla como {acronimos_str})")
                response_parts.append(f"Por favor, proporciona la información de nuevo: {get_formatted_question(campo_esperado)}")
                state['last_question_field'] = campo_esperado

    elif intent == "Saludo":
        if contexto['nombre_empresa'] is None and not user_name:
             extracted_name = extract_name_from_greeting(user_input)
             if extracted_name:
                  state['user_name'] = extracted_name;
        saludo_template = random.choice(respuestas_saludo)
        saludo_formateado = format_greeting(saludo_template, state['user_name'])
        response_parts.append(saludo_formateado.split('¿')[0].strip('.?! '))
        if not state['conversation_complete'] and state['last_question_field'] and contexto.get(state['last_question_field']) is None:
             response_parts.append(f"Continuemos. {get_formatted_question(state['last_question_field'])}")
        elif not state['conversation_complete']:
             campo_siguiente_key = get_next_question_key(contexto)
             if campo_siguiente_key:
                  response_parts.append(f"Continuemos. {get_formatted_question(campo_siguiente_key)}")
                  state['last_question_field'] = campo_siguiente_key

    elif intent == "Insulto":
        response_parts.append(random.choice(respuestas_insulto).split(':')[0].strip() + ".")
        if not state['conversation_complete'] and state['last_question_field'] and contexto.get(state['last_question_field']) is None:
             response_parts.append(get_formatted_question(state['last_question_field']))

    elif intent == "Pregunta Personal":
        response_parts.append(random.choice(respuestas_pregunta_personal).split(':')[0].strip() + ".")
        if not state['conversation_complete'] and state['last_question_field'] and contexto.get(state['last_question_field']) is None:
             response_parts.append(get_formatted_question(state['last_question_field']))

    elif intent == "Pregunta Económica":
         term_found = next((term for term in terminos_economicos if term in user_input.lower()), None)
         if term_found: response_parts.append(f"Claro, sobre '{term_found.capitalize()}': {terminos_economicos[term_found]}")
         else: response_parts.append("Entiendo tu duda económica, pero no identifiqué el término exacto. Puedes preguntarme por 'Activos', 'Pasivos', 'Cartera', etc.")
         if not state['conversation_complete'] and state['last_question_field'] and contexto.get(state['last_question_field']) is None:
              response_parts.append(f"Volviendo a lo nuestro... {get_formatted_question(state['last_question_field'])}")

    elif intent == "Pregunta sobre Proceso":
         field_topic_found = next((key for key in CAMPOS_REQUERIDOS if any(kw in user_input.lower() for kw in key.split('_'))) , None)
         if field_topic_found and field_topic_found in explicacion_proceso_campos:
              response_parts.append(f"Sobre '{field_topic_found.replace('_',' ')}': {explicacion_proceso_campos[field_topic_found]['explicacion']}")
         else:
              response_parts.append(f"Claro, te explico: {descripcion_general.split('.')[0]}. Necesito información como nombre de la empresa, sector, número de empleados, ingresos o activos, cartera y deudas.")
         if not state['conversation_complete'] and state['last_question_field'] and contexto.get(state['last_question_field']) is None:
              response_parts.append(f"Continuemos... {get_formatted_question(state['last_question_field'])}")

    else:         
        response_parts.append("No estoy seguro de entender eso.")
        if not state['conversation_complete']:
            current_field_to_ask = state.get('last_question_field')
            if current_field_to_ask and contexto.get(current_field_to_ask) is None:
                 response_parts.append(f"Podríamos continuar con: {get_formatted_question(current_field_to_ask)}")
            else:
                 campo_siguiente_key = get_next_question_key(contexto)
                 if campo_siguiente_key:
                      response_parts.append(f"Quizás podríamos continuar con: {get_formatted_question(campo_siguiente_key)}")
                      state['last_question_field'] = campo_siguiente_key
                 else:
                      state['conversation_complete'] = True
                      response_parts.append("Ya tengo toda la información. Procederé al análisis.")


    final_response = " ".join(response_parts)
    return final_response, state


def get_final_analysis(state):
    """Realiza el análisis final y devuelve el mensaje de resultado."""
    if get_next_question_key(state['contexto']) is None:
        lgbm_input_data = format_data_for_lgbm(state['contexto'])
        if lgbm_input_data is not None:
            resultado_final_label = predict_lgbm(lgbm_input_data)
            respuestas_para_resultado = plantillas_resultado_final.get(resultado_final_label, ["No pude determinar una clasificación final con los datos proporcionados."])
            respuesta_final_elegida = random.choice(respuestas_para_resultado)
            resultado = f"#### ✅💰 Resultado del Análisis Financiero Preliminar \n\n{respuesta_final_elegida}\n\n"
            resultado += random.choice(respuestas_despedida)
            return resultado
        else:
            error_msg = "Hubo un problema al preparar los datos para el análisis final. No puedo darte una clasificación."
            error_msg += f"\n{random.choice(respuestas_despedida)}"
            return error_msg
    else:
        incomplete_msg = "Aún falta información para realizar el análisis."
        incomplete_msg += f"\n{random.choice(respuestas_despedida)}"
        return incomplete_msg

