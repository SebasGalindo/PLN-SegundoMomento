import streamlit as st

st.set_page_config(page_title="Explicación: chatbot_logic.py", layout="wide")

st.title("Análisis del Script: `chatbot_logic.py` (Lógica Central del Chatbot)")

st.header("Propósito General y Rol en el Sistema")

st.markdown("""
El script `chatbot_logic.py` funciona como el **motor conversacional central** del sistema. Se encarga de gestionar el diálogo, interpretar las entradas del usuario utilizando diversos modelos y técnicas de Procesamiento de Lenguaje Natural (PLN), extraer información clave, mantener el estado de la conversación y, eventualmente, invocar el análisis final basado en los datos recopilados. Su diseño modular permite la interacción con una interfaz de usuario externa (como Streamlit) mediante el intercambio de mensajes y estados.

Este script integra varios componentes esenciales: los modelos de **Intención** y **NER** (basados en BERT/BETO), un modelo de **Clasificación Final** (LightGBM), la librería **spaCy** para análisis lingüístico detallado (similitud, parsing numérico, análisis de confirmación), y **recursos de datos** (JSON, archivos de texto) que contienen plantillas y configuraciones.
""")

st.header("Desglose Detallado del Código y Flujo Operativo")

st.subheader("1. Configuración e Importaciones (Sección 1)")
st.markdown("""
Se realizan las importaciones de librerías (`json`, `os`, `random`, `re`, `joblib`, `spacy`, `torch`, `numpy`, `pandas`, `transformers`, etc.) y se definen constantes globales. Estas incluyen rutas a modelos y datos, identificadores de modelos (`MODELO_INTENCION`, `MODELO_NER`, `MODELO_SPACY`), parámetros de configuración (`MAX_LEN_BERT`, `DEFAULT_CURRENCY`, `SIMILARITY_THRESHOLD`), la lista de campos de información requeridos (`CAMPOS_REQUERIDOS`), el mapeo de campos a etiquetas NER (`CAMPO_A_NER_LABEL`), y listas de lemas para análisis de confirmación (`LEMAS_POSITIVOS`, `LEMAS_NEGATIVOS`).
""")
# Código omitido por brevedad

st.subheader("2. Carga de Recursos (Sección 2 - Ejecutada una sola vez)")
st.markdown("""
Esta sección es fundamental para la eficiencia. **Se ejecuta una única vez** al inicio (cuando se importa el módulo), cargando todos los modelos y datos pesados en memoria para un acceso rápido durante la conversación:
* **Datos JSON y TXT:** Carga plantillas de preguntas/respuestas, configuraciones, términos económicos y categorías de empresa.
* **Modelo de Intención (BERT/BETO):** Carga el tokenizador (`AutoTokenizer`) y el modelo (`AutoModelForSequenceClassification`) afinado para clasificación de secuencias.
* **Modelo NER (BERT/BETO):** Carga el tokenizador (`AutoTokenizer`) y el modelo (`AutoModelForTokenClassification`) afinado para reconocimiento de entidades nombradas.
* **Modelo LightGBM:** Carga el modelo (`lgbm_model`), el codificador de etiquetas (`lgbm_label_encoder`) y las features esperadas (`lgbm_features`) desde el archivo `.joblib`.
* **Modelo spaCy (`nlp`):** Carga el modelo `es_core_news_lg`. Este modelo es crucial porque proporciona varias capacidades de PLN utilizadas directamente en este script:
    * **Tokenización a nivel de palabra:** Usada como paso previo para otras operaciones de spaCy.
    * **Lematización:** Obtención de la forma base de las palabras.
    * **POS Tagging:** Identificación de la categoría gramatical (aunque no se use explícitamente su *output* en este script, el análisis subyacente ayuda).
    * **Embeddings:** Vectores pre-entrenados para palabras y documentos, usados para calcular similitud semántica.
* **Embeddings de Categorías:** Calcula y almacena los embeddings spaCy para las categorías de empresa leídas del archivo TXT.
* **Tasas de Cambio:** Obtiene tasas de yfinance o usa valores por defecto.

Esta carga inicial asegura que el chatbot pueda responder rápidamente sin recargar modelos en cada turno.
""")
# Código omitido por brevedad

st.subheader("3. Funciones Auxiliares (Sección 3) y Uso de Técnicas PLN")
st.markdown("""
Se definen funciones modulares para tareas específicas, varias de las cuales aplican técnicas de PLN:

* **`predict_intent(text)` y `predict_ner(text)`:** Estas funciones encapsulan el uso de los modelos BERT afinados.
    * **Tokenización (BERT):** Internamente, estas funciones utilizan los `AutoTokenizer` cargados (`intent_tokenizer`, `ner_tokenizer`) para realizar la **tokenización por subpalabras (WordPiece)** requerida por BERT/BETO. Este proceso convierte el texto en secuencias de IDs numéricos que el modelo puede procesar.
* **`parse_numero(texto)`:** Extrae valores numéricos y monedas de texto potencialmente complejo (ej: "mil quinientos millones de pesos").
    * **Tokenización (spaCy):** Utiliza `nlp(texto)` para dividir la entrada en palabras y símbolos.
    * **Lematización (spaCy):** Accede a `token.lemma_` para normalizar palabras numéricas (ej: 'millones' -> 'millon') y compararlas con listas conocidas (`numeros_texto_spacy`, `multiplicadores_spacy`). **Utilidad:** Permite reconocer números expresados textualmente de forma robusta.
    * **Análisis Lingüístico (spaCy):** Usa `token.like_num` para identificar tokens que parecen números, incluso si no son estrictamente dígitos.
* **`encontrar_mejor_categoria(texto_categoria)`:** Busca la categoría de empresa más similar a la entrada del usuario.
    * **Tokenización y Embeddings (spaCy):** Usa `nlp(texto_categoria)` para procesar el texto y obtener su vector de embedding contextualizado. La calidad de este embedding depende de la tokenización y el análisis realizado por spaCy. Compara este vector con los embeddings precalculados de las categorías conocidas mediante similitud coseno. **Utilidad:** Permite encontrar coincidencias semánticas incluso si el usuario no usa el nombre exacto de la categoría.
* **`check_confirmation(text)`:** Determina si una respuesta es afirmativa, negativa o ambigua.
    * **Tokenización (spaCy):** Usa `nlp(texto)` para obtener los tokens.
    * **Lematización (spaCy):** Itera sobre los tokens y utiliza `token.lemma_` para obtener la forma base de cada palabra. Compara estos lemas con `LEMAS_POSITIVOS` y `LEMAS_NEGATIVOS`. **Utilidad:** Es crucial para la robustez. Permite reconocer "sí", "si", "correcto", "correcta", "afirmativo" (y sus variantes) como confirmaciones, y "no", "incorrecto", "falso" (y sus variantes) como negaciones, independientemente de la forma exacta utilizada por el usuario.
    * **(POS Tagging - Implícito):** Aunque el código no accede directamente a `token.pos_`, el análisis gramatical que realiza spaCy para determinar el lema correcto a menudo se apoya en el POS tag del token.
* **Otras funciones:** `get_next_question_key`, `get_formatted_question`, `format_greeting`, `convert_to_COP`, `format_data_for_lgbm`, `predict_lgbm`, `extract_name_from_greeting` realizan tareas de gestión de flujo, formato o aplicación de reglas/modelos sin un uso directo destacado de tokenización, POS tagging o lematización más allá de lo que usan las funciones anteriores.
""")
# Código omitido por brevedad


st.subheader("4. Funciones Principales y Lógica Conversacional (Sección 4)")
st.markdown("""
Esta sección contiene la lógica central que maneja la conversación turno a turno.

* **`get_initial_state()` y `get_initial_message(state)`:** Preparan el inicio de la conversación, estableciendo el estado inicial y formulando la primera pregunta.
* **`process_user_input(user_input, current_state)`:** Es el corazón del procesamiento. El flujo paso a paso, destacando el uso de PLN, es:
    1.  Recibe `user_input` y `current_state`.
    2.  **Manejo de Confirmación:**
        * Si el estado indica `confirmation_pending`:
            * Usa `check_confirmation(user_input)`. Esta función internamente aplica **Tokenización (spaCy)** y **Lematización (spaCy)** para interpretar la respuesta sí/no del usuario.
            * Si es 'yes', actualiza el contexto y prepara la siguiente pregunta o finaliza.
            * Si es 'no', prepara una pregunta para corregir el dato.
            * Si es 'unclear', repite la pregunta de confirmación.
    3.  **Manejo de Nueva Entrada (sin confirmación pendiente):**
        * **Clasificación de Intención:** Llama a `predict_intent(user_input)`. Esta función usa el **modelo BERT de Intención**, el cual realiza **Tokenización (BERT subword)** internamente.
        * **Intenciones Simples:** Maneja saludos, insultos, preguntas, etc., usando respuestas predefinidas. `extract_name_from_greeting` usa reglas.
        * **Intención de Respuesta/Corrección:**
            * Identifica el `campo_esperado`.
            * **Extracción NER:** Llama a `predict_ner(user_input)`. Esta función usa el **modelo BERT NER**, que también realiza **Tokenización (BERT subword)** interna y predice etiquetas BIO. La función reconstruye las entidades (texto, etiqueta, posición).
            * **Validación/Parsing:**
                * Si NER encontró entidades relevantes:
                    * Para `area_categoria`, usa `encontrar_mejor_categoria` (que aplica **Tokenización y Embeddings spaCy**) para validar contra la lista conocida.
                    * Para campos numéricos, usa `parse_numero` sobre el texto de la entidad (aplicando **Tokenización y Lematización spaCy**, además de reglas).
                * Si NER falló o el parsing falló, intenta `parse_numero` o `encontrar_mejor_categoria` sobre el `user_input` completo como fallback.
            * **Preparación de Confirmación:** Si se obtiene un dato válido (o se confirma texto crudo), se establece `confirmation_pending` en el estado y se formula la pregunta "¿Es X correcto para Y? (Sí/No)".
            * Si falla, se pide la información de nuevo.
        * **Intención Desconocida:** Se da una respuesta genérica.
    4.  **Devolución:** Retorna la respuesta textual del bot y el diccionario de estado actualizado.
* **`get_final_analysis(state)`:**
    * Se llama cuando todos los datos están completos.
    * Prepara los datos con `format_data_for_lgbm`.
    * Usa `predict_lgbm` para obtener la clasificación final del modelo LightGBM.
    * Formatea y devuelve el mensaje de análisis y despedida.
""")

st.header("Ejemplo de Flujo Simplificado")

st.markdown("""
Imaginemos el siguiente intercambio:

1.  **Chatbot (Estado Inicial):** "¡Hola! Soy tu asistente financiero. ¿Podrías indicarme el nombre de tu empresa?" (`last_question_field` = 'nombre_empresa', `contexto` vacío)
2.  **Usuario:** "Claro, es TechSoluciones Innovadoras Ltda"
3.  **`process_user_input` ("Claro, es TechSoluciones Innovadoras Ltda", estado_anterior):**
    * `confirmation_pending` es `None`.
    * `predict_intent` (BERT) -> "Respuesta Nombre".
    * `campo_esperado` = 'nombre_empresa'.
    * `predict_ner` (BERT) -> Detecta `[('TechSoluciones Innovadoras Ltda', 'NOMBRE_EMPRESA', start, end)]`.
    * Validación OK. `dato_validado` = "TechSoluciones Innovadoras Ltda".
    * Se establece `confirmation_pending` = `{'field': 'nombre_empresa', 'validated_value': 'TechSoluciones Innovadoras Ltda', ...}`.
    * **Respuesta:** "¿Es TechSoluciones Innovadoras Ltda correcto para nombre empresa? (Sí/No)"
    * **Devuelve:** (Respuesta, estado_con_confirmacion_pendiente)
4.  **Usuario:** "sip"
5.  **`process_user_input` ("sip", estado_con_confirmacion_pendiente):**
    * `confirmation_pending` **no** es `None`.
    * `check_confirmation("sip")` (spaCy Tokenize + Lematize -> 'sí') -> 'yes'.
    * Se actualiza `contexto['nombre_empresa']` = "TechSoluciones Innovadoras Ltda".
    * Se limpia `confirmation_pending`.
    * `get_next_question_key` -> 'area_categoria'.
    * `get_formatted_question` -> "¿A qué área o categoría pertenece la empresa? (Ej: Consultoría Tecnológica)"
    * **Respuesta:** "Ok, entendido nombre empresa. Continuemos. ¿A qué área o categoría pertenece la empresa? (Ej: Consultoría Tecnológica)"
    * Se actualiza `last_question_field` = 'area_categoria'.
    * **Devuelve:** (Respuesta, estado_actualizado)
6.  ... (La conversación continúa de forma similar, usando NER, parse_numero, encontrar_mejor_categoria, check_confirmation, etc., hasta llenar todos los `CAMPOS_REQUERIDOS`) ...
7.  **`process_user_input` (última confirmación "sí"):**
    * ... (Actualiza último campo en `contexto`) ...
    * `get_next_question_key` -> `None`.
    * Marca `conversation_complete` = `True`.
    * **Respuesta:** "¡Excelente! Ya tengo toda la información necesaria. Permíteme analizarla..."
    * **Devuelve:** (Respuesta, estado_final_con_complete_True)
8.  **Frontend (Streamlit):** Detecta `state['conversation_complete'] == True`. Llama a `get_final_analysis(estado_final)`.
9.  **`get_final_analysis`:**
    * `format_data_for_lgbm` -> Prepara DataFrame.
    * `predict_lgbm` -> Obtiene 'Nivel Económico'.
    * Formatea respuesta final.
    * **Devuelve:** "=== Resultado del Análisis... ===\n\n[Resultado basado en predicción LightGBM]\n\n¡Adiós!..."

Este flujo ilustra cómo el script combina la comprensión de la intención, la extracción de entidades, la validación basada en PLN y la gestión de estado para guiar la conversación hacia su objetivo.
""")


st.header("Conclusión")

st.success("""
`chatbot_logic.py` es el componente orquestador que da vida al chatbot. Integra de manera efectiva modelos avanzados de deep learning (BERT para Intención y NER) con técnicas clásicas y robustas de PLN (spaCy para lematización, parsing numérico, embeddings de similitud) y un modelo de ML tradicional (LightGBM) para la clasificación final. La gestión cuidadosa del estado conversacional y el uso estratégico de la tokenización (tanto subword para BERT como word-level para spaCy), la lematización (especialmente para confirmaciones y números) y los embeddings (para similitud de categorías) le permiten mantener un diálogo coherente, extraer información precisa y alcanzar el objetivo de análisis financiero preliminar.
""")
