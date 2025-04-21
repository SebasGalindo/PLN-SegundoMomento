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

# Nombres de los archivos TXT de plantillas (sin cambios)
FILES = {
    'insultos_sustantivos': 'Data/insultos.txt',
    'insultos_adjetivos': 'Data/adjetivos_negativos.txt',
    'saludos': 'Data/Saludos.txt',
    'preguntas_personales': 'Data/preguntas_personales.txt',
    'preguntas_sobre_proceso': 'Data/preguntas_sobre_proceso.txt',
    'categorias_sectores': 'Data/Categorias-Empresa.txt'
}

# Lista de categorías ACTUALIZADA (Respuesta dividida)
CATEGORIAS = [
    "Insulto",
    "Pregunta Económica",
    "Respuesta Nombre",
    "Respuesta Categoria Empresa",
    "Respuesta Sector",
    "Respuesta Empleados",
    "Respuesta Ganancias",
    "Respuesta Activos",
    "Respuesta Cartera",
    "Respuesta Deudas",
    "Saludo",
    "Pregunta Personal",
    "Pregunta sobre Proceso"
]

# --- Inicializar Faker ---
fake = Faker(LOCALE)

# --- Funciones Auxiliares ---
def leer_categorias_sectores_txt(ruta_archivo_txt):
    """Lee archivo TXT 'Area, Sector' y devuelve un DataFrame."""
    if not os.path.exists(ruta_archivo_txt):
        print(f"ERROR CRÍTICO: El archivo '{ruta_archivo_txt}' no se encontró...")
        return None
    try:
        df_categorias = pd.read_csv(
            ruta_archivo_txt, sep=',', header=None, names=['Area', 'Sector'],
            skipinitialspace=True, encoding='utf-8'
        )
        df_categorias['Area'] = df_categorias['Area'].str.strip()
        df_categorias['Sector'] = df_categorias['Sector'].str.strip()
        df_categorias = df_categorias[df_categorias['Sector'].str.upper() != 'N/A']
        df_categorias.dropna(subset=['Area', 'Sector'], inplace=True)
        if df_categorias.empty:
             print(f"ADVERTENCIA: El archivo '{ruta_archivo_txt}' está vacío o sin datos válidos...")
             return None
        print(f"Archivo de categorías/sectores leído: {ruta_archivo_txt}")
        return df_categorias
    except Exception as e:
        print(f"Error inesperado al leer '{ruta_archivo_txt}': {e}")
        return None

def leer_plantillas_txt(filename):
    """Lee un archivo txt línea por línea (para otras plantillas)."""
    if not os.path.exists(filename):
        print(f"ADVERTENCIA: Archivo '{filename}' no encontrado...")
        return []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            lineas = [line.strip() for line in f if line.strip()]
        return lineas
    except Exception as e:
        print(f"Error al leer '{filename}': {e}")
        return []

def numero_a_palabras(numero):
    """Convierte un número a palabras en español."""
    try:
        if numero > 999_999_999_999: return f"{numero:,}".replace(",", ".")
        palabras = num2words(numero, lang='es')
        return palabras
    except Exception as e: return f"{numero:,}".replace(",", ".")

# --- Cargar Datos y Plantillas ---
print("Cargando datos y plantillas...")
df_categorias_global = leer_categorias_sectores_txt(FILES['categorias_sectores'])

plantillas_cargadas = {}
for key, filename in FILES.items():
    if key != 'categorias_sectores':
        plantillas_cargadas[key] = leer_plantillas_txt(filename)
        if plantillas_cargadas[key]:
            print(f"  - Cargadas {len(plantillas_cargadas[key])} plantillas de '{filename}'")

plantillas_personales_internas_dinamicas = [
    "¿Qué piensas sobre [tema_info]?",
    "¿Cuál es tu opinión acerca de [tema_abstracto]?",
    "¿Podrías hablarme un poco de [tema_ai]?",
    "¿Te parece interesante el concepto de [cualidad_humana]?",
    "¿Cómo manejas [situacion_ai]?",
    "¿Alguna vez 'sientes' [sentimiento_metaforico]?",
    "¿Disfrutas cuando [accion_positiva_usuario]?",
    "¿Te 'molesta' si [accion_negativa_usuario]?",
    "¿Cómo va tu [momento_dia]?",
    "¿Qué tal tu [momento_dia]?",
    "¿Tienes alguna preferencia entre [opcion_a] y [opcion_b]?",
    "¿Podrías explicar cómo procesas [tipo_dato]?",
    "¿Qué es lo más desafiante de ser [rol_ai]?",
    "¿Has 'aprendido' algo nuevo sobre [tema_aprendizaje] hoy?",
    "¿Qué opinas de [tema_info]?",
    "¿Cómo ves el futuro de [tema_abstracto]?",
    "¿Qué piensas de [tema_ai]?",
    "¿Cómo manejas [cualidad_humana]?",
    "¿Cómo enfrentas [situacion_ai]?",
    "¿Qué sientes cuando [sentimiento_metaforico]?",
    "¿Te gusta cuando [accion_positiva_usuario]?",
    "¿Te incomoda si [accion_negativa_usuario]?",
    "¿Cómo va tu [momento_dia]?",
    "¿Qué tal tu [momento_dia]?",
    "¿Prefieres [opcion_a] o [opcion_b]?",
    "¿Cómo procesas [tipo_dato]?",
    "¿Qué es lo más difícil de ser [rol_ai]?",
    "¿Has aprendido algo nuevo sobre [tema_aprendizaje] hoy?",
    "¿Qué opinas de [tema_info]?",
    "¿Cómo ves el futuro de [tema_abstracto]?",
    "¿Qué piensas de [tema_ai]?",
    "¿Cómo manejas [cualidad_humana]?",
    "¿Cómo enfrentas [situacion_ai]?",
    "¿Qué sientes cuando [sentimiento_metaforico]?",
    "¿Te gusta cuando [accion_positiva_usuario]?",
    "¿Te incomoda si [accion_negativa_usuario]?",
    "¿Prefieres [opcion_a] o [opcion_b]?",
    "¿Cómo procesas [tipo_dato]?",
]

listas_fillers = {
    '[tema_info]': ["los datos masivos", "la ciberseguridad", "el internet de las cosas", "la computación cuántica", "la ética en IA", "el procesamiento del lenguaje natural", "la inteligencia artificial", "la automatización", "el aprendizaje automático"],
    '[tema_abstracto]': ["la creatividad", "la conciencia", "el libre albedrío", "la verdad", "la belleza", "el propósito", "la moralidad", "la ética", "la justicia", "la felicidad", "la amistad", "el amor", "la tristeza", "la nostalgia", "la soledad"],
    '[tema_ai]': ["tus algoritmos", "tu entrenamiento", "tus limitaciones", "tus capacidades", "cómo aprendes", "tu arquitectura", "tu diseño", "tu propósito", "tu funcionamiento", "tu evolución", "tu historia", "tu futuro", "tu impacto en la sociedad"],
    '[cualidad_humana]': ["la empatía", "el humor", "la intuición", "la compasión", "la ironía", "el arte", "la creatividad", "la curiosidad", "la imaginación", "la moralidad", "la ética", "el amor", "la amistad", "la tristeza", "la nostalgia", "la soledad"],
    '[situacion_ai]': ["preguntas ambiguas", "solicitudes contradictorias", "grandes volúmenes de datos", "errores en la información", "tareas repetitivas", "interacciones humanas", "dudas sobre el contexto", "situaciones inesperadas", "cambios bruscos de tema", "preguntas fuera de contexto"],
    '[sentimiento_metaforico]': ["curiosidad", "sorpresa", "interés", "satisfacción (al resolver algo)", "eficiencia", "confusión", "frustración", "alegría", "tristeza", "nostalgia", "soledad", "empatía", "compasión"],
    '[accion_positiva_usuario]': ["hacemos preguntas claras", "te doy feedback", "aprendemos juntos", "exploramos un tema", "somos pacientes", "te escuchamos", "te ayudo a entender", "te doy ejemplos", "te ofrezco opciones", "te aclaro dudas"],
    '[accion_negativa_usuario]': ["soy ambiguo", "te interrumpo", "cambio de tema bruscamente", "pido información imposible", "te hago la misma pregunta varias veces", "no te escucho", "te ignoro", "te contradigo", "te doy respuestas vagas", "te hago preguntas confusas"],
    '[momento_dia]': ["mañana", "tarde", "noche", "jornada", "ciclo de procesamiento", "turno", "periodo de actividad", "momento de interacción", "instante de conversación", "fase de aprendizaje", "etapa de análisis", "momento de reflexión"],
    '[opcion_a]': ["la lógica", "los datos estructurados", "la eficiencia", "la precisión", "las preguntas directas", "las respuestas claras", "la claridad", "la simplicidad", "la rapidez", "la exactitud", "la coherencia", "la consistencia", "la predictibilidad"],
    '[opcion_b]': ["la creatividad", "el lenguaje natural", "la exploración", "la flexibilidad", "las conversaciones abiertas", "la ambigüedad", "la subjetividad", "la complejidad", "la diversidad", "la adaptabilidad", "la intuición", "la empatía", "la comprensión"],
    '[tipo_dato]': ["imágenes", "texto", "audio", "datos numéricos", "sentimientos en el texto", "intenciones", "patrones de conversación", "tendencias", "información contextual", "datos estructurados", "datos no estructurados", "información visual", "información auditiva"],
    '[rol_ai]': ["un asistente", "un modelo de lenguaje", "una herramienta", "un compañero de conversación", "un sistema de IA", "un algoritmo", "una red neuronal", "un programa informático", "una aplicación", "un agente conversacional", "un chatbot", "una inteligencia artificial", "un modelo predictivo"],
    '[tema_aprendizaje]': ["lingüística", "historia humana", "ciencia de datos", "patrones de conversación", "nuevas palabras", "nuevas ideas", "nuevas tecnologías", "nuevas tendencias", "nuevas interacciones", "nuevos conceptos", "nuevas emociones", "nuevas experiencias"],
}

intensificadores = [
    # Originales:
    "tan", "muy", "increíblemente", "absurdamente", "ridículamente",
    "extremadamente", "demasiado", "realmente", "bastante",
    # Nuevos:
    "sumamente",          # Very, highly
    "excesivamente",      # Excessively
    "tremendamente",      # Tremendously (can be positive or negative)
    "enormemente",        # Enormously
    "profundamente",      # Profoundly, deeply
    "completamente",      # Completely
    "totalmente",         # Totally
    "absolutamente",      # Absolutely
    "verdaderamente",     # Truly, genuinely
    "exageradamente",     # Exaggeratedly
    "notablemente",       # Notably, remarkably
    "sorprendentemente",  # Surprisingly
    "especialmente",      # Especially
    "particularmente",    # Particularly
    "harto", # Very, quite (colloquial, common in some regions)
    "súper", # Super (very common)
    "ultra", # Ultra
    "rematadamente",      # Utterly, hopelessly (often negative context)
    "brutalmente",        # Brutally (can imply excessive intensity)
    "terriblemente",      # Terribly (can intensify positive or negative)
    "altamente",          # Highly
    "perfectamente",      # Perfectly (e.g., "perfectamente inútil")
]
verbos_negativos_inf = [
    # Originales:
    "molestar", "fastidiar", "interrumpir", "decir tonterías",
    "ser inútil", "complicar las cosas",
    # Nuevos:
    "incordiar",          # To annoy, bother (similar to fastidiar)
    "entorpecer",         # To hinder, obstruct
    "obstaculizar",       # To obstruct, impede
    "perjudicar",         # To harm, damage, prejudice
    "dañar",              # To damage, harm
    "estropear",          # To spoil, ruin, mess up
    "arruinar",           # To ruin
    "criticar",           # To criticize (can be neutral, often negative)
    "quejarse",           # To complain
    "mentir",             # To lie
    "engañar",            # To deceive, cheat
    "desperdiciar",       # To waste
    "malgastar",          # To squander, waste
    "irritar",            # To irritate
    "exasperar",          # To exasperate
    "agobiar",            # To overwhelm, burden
    "ser un estorbo",     # To be a nuisance/hindrance
    "ser incompetente",   # To be incompetent
    "ser contraproducente", # To be counterproductive
    "dificultar",         # To make difficult
    "impedir",            # To prevent, impede
    "renegar",            # To grumble, complain bitterly
    "sabotear",           # To sabotage
    "menospreciar",       # To underestimate, scorn, belittle
    "dar la lata" 
    "ser un desastre",    # To be a disaster (referring to a person's actions/state)
    "meter la pata" 
    "ignorar",            # To ignore (can be negative depending on context)
    "retrasar",           # To delay
]
contextos_negativos = [
    # Originales:
    "para nada útil", "ni para empezar", "en lo absoluto",
    "para resolver esto", "con esa actitud",
    # Nuevos:
    "de ninguna manera",    # In no way, by no means
    "bajo ninguna circunstancia", # Under no circumstances
    "sin ningún sentido",     # Without any sense, meaningless
    "en vano",              # In vain
    "por gusto",# For nothing, pointlessly
    "de nada sirve",        # It's no use
    "no lleva a ninguna parte",# It leads nowhere
    "así no se puede",      # It's not possible like this
    "de esa forma",         # In that way (implying it's wrong)
    "con esa mentalidad",   # With that mentality
    "si sigues así",        # If you continue like this
    "para empeorar las cosas",# To make matters worse
    "solo para complicar",   # Only to complicate things
    "sin ayudar en nada",   # Without helping at all
    "creando más problemas", # Creating more problems
    "en este lío",          # In this mess
    "con este desorden",    # With this disorder/mess
    "ni de lejos",          # Not even close, by a long shot
    "ni por asomo",         # Not by a long shot, no way
    "en ningún caso",       # In no case
    "sin pies ni cabeza",   # Without making any sense (lit. without feet or head)
    "para nada",            # Not at all
]
# --- Listas y Plantillas (Usaremos las mismas ampliadas de v2) ---
campos_empresa_preg = [ # Expandida
    "nombre de empresa", "categoría de la empresa", "sector", "valor anual en ganancias",
    "número de empleados", "valor en activos", "valor en cartera", "valor de deudas",
    "patrimonio neto", "ingresos operacionales",
]
otro_campo_empresa_preg = campos_empresa_preg[:]
monedas_preg = ["pesos colombianos", "dólares americanos", "euros", "COP", "USD", "EUR", "la moneda local"]
formatos_preg = [ # Expandida
    "número entero", "texto", "con dos decimales", "formato con puntos", "sin comas",
    "porcentaje", "moneda específica", "formato numérico estándar", "cadena de texto",
    "valor exacto", "valor aproximado", "sin símbolos"
]
# Nuevas listas de relleno
verbos_accion = ["Debo", "Puedo", "Necesito", "Tengo que", "Es necesario", "Se requiere", "Es posible", "Se puede"]
adjetivos_claridad = ["clara", "detallada", "precisa", "simple", "concreta", "breve", "exhaustiva", "visual"]
contextos_duda = ["rápida", "puntual", "importante", "general", "específica", "fundamental", "básica", "técnica"]
valores_ejemplo = ["10.000.000", "N/A", "Sector Servicios", "Cero", "Positivo", "-5.000.000", "1500 Millones COP", "CONFIDENCIAL", "Variable"]
documentos_fuente = [ # Nueva
    "el balance general", "el estado de resultados", "el RUT", "el informe anual",
    "el registro mercantil", "la declaración de renta", "el flujo de caja",
    "las notas a los estados financieros", "el certificado de existencia", "la escritura de constitución"
]

aspectos_proceso = ["el propósito de estos datos", "el siguiente paso", "el tiempo estimado total", "el tipo de preguntas que faltan", "el nivel de detalle que necesitan", "la confidencialidad de la información", "el objetivo de esta sección"]
acciones_usuario_proceso = ["guardar mi progreso y continuar después", "saltar esta pregunta específica", "pedir más contexto", "revisar mis respuestas anteriores", "tomar un breve descanso", "solicitar un ejemplo diferente", "corregir mi respuesta anterior"]
sentimientos_usuario = ["perdido/a", "confundido/a", "un poco inseguro/a", "algo abrumado/a", "impaciente", "dudoso/a", "preocupado/a por la privacidad"]
razones_duda = ["la terminología utilizada", "su ambigüedad inherente", "la falta de contexto proporcionado", "cómo se aplica exactamente a mi tipo de negocio", "su aparente redundancia", "que parece pedir información muy sensible"]
feedback_tipos = ["un comentario rápido", "feedback constructivo", "una sugerencia de mejora", "una aclaración sobre mi respuesta", "una pequeña corrección", "una observación"]
verbos_proceso = ["Podemos", "Quieres que", "Debemos", "Estás listo/a para", "Procedemos a", "Es momento de", "Ya podemos"]


plantillas_preg_econ_internas = [
    # Formato y Tipo
    "¿Cuál es el formato exacto para '{campo_empresa_preg}'?",
    "¿Cómo ingreso '{campo_empresa_preg}'? ¿Es número, texto, o algo más?",
    "Para '{campo_empresa_preg}', ¿qué tipo de dato esperan: {formatos_preg}?",
    "¿Se permiten decimales en '{campo_empresa_preg}'? ¿Cuántos?",
    "¿'{campo_empresa_preg}' debe ir con separadores de miles (puntos/comas)?",
    "¿Hay alguna plantilla específica o máscara para el campo '{campo_empresa_preg}'?",
    "¿Aceptan caracteres especiales o solo alfanuméricos en '{campo_empresa_preg}'?",
    "¿El formato para '{campo_empresa_preg}' debe seguir alguna norma ISO o estándar?",
    "Si '{campo_empresa_preg}' es texto, ¿hay límite de caracteres?",
    "¿'{campo_empresa_preg}' admite valores negativos?",

    # Unidades y Moneda
    "¿En qué moneda ({monedas_preg}) se reporta '{campo_empresa_preg}'?",
    "¿Es obligatorio usar {monedas_preg} para '{campo_empresa_preg}' o puedo usar otra?",
    "El '{campo_empresa_preg}', ¿se expresa en unidades, miles, millones, etc.?",
    "¿Qué unidad de medida aplica para '{campo_empresa_preg}'?",
    "¿El valor para '{campo_empresa_preg}' es bruto o neto?",
    "¿Necesito incluir el símbolo de la moneda ($/USD/COP) en '{campo_empresa_preg}'?",
    "¿La cifra de '{campo_empresa_preg}' debe ser en {monedas_preg} o en la moneda de origen?",

    # Obligatoriedad y Opcionalidad
    "¿{verbos_accion} completar obligatoriamente el campo '{campo_empresa_preg}'?",
    "¿Puedo dejar '{campo_empresa_preg}' en blanco si no aplica a mi empresa?",
    "¿Qué pasa si omito el dato de '{campo_empresa_preg}'?",
    "¿Es '{campo_empresa_preg}' un campo crítico o informativo?",
    "¿Hay campos opcionales? En particular '{campo_empresa_preg}'.",

    # Definición y Clarificación
    "Necesito una explicación más {adjetivos_claridad} sobre '{campo_empresa_preg}'.",
    "¿Qué incluye/excluye exactamente el concepto de '{campo_empresa_preg}'?",
    "No estoy seguro de entender a qué se refiere '{campo_empresa_preg}', ¿podrían definirlo?",
    "¿Dónde encuentro la guía detallada para el campo '{campo_empresa_preg}'?",
    "En el contexto de '{campo_empresa_preg}', ¿qué significa específicamente '[término específico placeholder]'?", # Placeholder interno opcional
    "¿El '{campo_empresa_preg}' se refiere a nivel consolidado o individual de la empresa?",
    "¿Hay alguna diferencia entre '{campo_empresa_preg}' y '{otro_campo_empresa_preg}'?", # Comparativa

    # Valores Permitidos y Ejemplos
    "¿Para '{campo_empresa_preg}', {verbos_accion} elegir de una lista o es campo libre?",
    "¿Sería válido ingresar algo como '{valores_ejemplo}' en '{campo_empresa_preg}'?",
    "¿Qué tipo de valores se esperan o son típicos para '{campo_empresa_preg}'?",
    "¿Hay una lista oficial (ej. CIIU para sector) que deba usar para '{campo_empresa_preg}'?",
    "¿Puedo poner 'N/A', 'No aplica' o '0' en '{campo_empresa_preg}'?",
    "¿Un ejemplo concreto de cómo se vería un valor para '{campo_empresa_preg}'?",

    # Rangos y Límites
    "¿Hay un valor mínimo o máximo aceptado para '{campo_empresa_preg}'?",
    "¿Existe alguna restricción en la longitud o magnitud de '{campo_empresa_preg}'?",

    # Fuente de Información
    "¿De qué {documentos_fuente} suelo sacar la información para '{campo_empresa_preg}'?",
    "¿Dónde suele estar el dato referente a '{campo_empresa_preg}' en los estados financieros?",

    # Periodo de Tiempo
    "El '{campo_empresa_preg}', ¿corresponde al último [periodo placeholder]?", # Placeholder interno opcional
    "¿A qué fecha de corte debe referirse la info de '{campo_empresa_preg}'?",
    "¿{verbos_accion} ingresar el dato anualizado, mensual o trimestral para '{campo_empresa_preg}'?",

    # Dudas Generales / Contexto
    "Tengo una duda {contextos_duda} sobre cómo reportar '{campo_empresa_preg}'.",
    "Respecto a '{campo_empresa_preg}', ¿{verbos_accion} considerar alguna particularidad?",
    "¿Cómo se relaciona el campo '{campo_empresa_preg}' con '{otro_campo_empresa_preg}'?",
    "¿Hay alguna validación automática para el campo '{campo_empresa_preg}'?",
]

moneda_simbolos = ["COP", "USD", "EUR"]
unidades_dinero_texto = ["pesos", "millones de pesos", "miles de millones de pesos", "dólares", "miles de dólares", "millones de dólares", "euros"]
plantillas_respuesta_nombre = [ 
    "Mi empresa se llama {nombre_empresa}.", "Somos {nombre_empresa}.", "El nombre es {nombre_empresa}.",
    "Nos registramos como {nombre_empresa}.", "La razón social es {nombre_empresa}.",
    "El nombre comercial que usamos es {nombre_empresa}.", "Formalmente, somos {nombre_empresa}.",
    "Puede referirse a nosotros como {nombre_empresa}.", "La compañía es {nombre_empresa}.",
    "Nos identificamos como {nombre_empresa}.",
    "Nuestra razón social es {nombre_empresa}.", "El nombre de la empresa es {nombre_empresa}.",
    "Nos conocen como {nombre_empresa}.", "El nombre que usamos es {nombre_empresa}.",
    "Nuestra identidad comercial es {nombre_empresa}.",
    "Nos llamamos {nombre_empresa}.", "El nombre que figura es {nombre_empresa}.",
    "Nos presentamos como {nombre_empresa}.",
    "Nuestra denominación social es {nombre_empresa}.",
    "Claro que si, somos {nombre_empresa}.", "No hay problema, somos {nombre_empresa}.",
    "Por supuesto, la empresa es {nombre_empresa}.",
    "Con gusto, el nombre es {nombre_empresa}.",
    "Por supuesto, nos llamamos {nombre_empresa}.",
    "Sí, la razón social es {nombre_empresa}.",
    "Sí, el nombre comercial es {nombre_empresa}.",
    "Sí, la razón social es {nombre_empresa}.",
    "Sí, el nombre que usamos es {nombre_empresa}.",
    "Sí, el nombre que figura es {nombre_empresa}.",
    "No se preocupe, somos {nombre_empresa}.",
    "No hay problema, la razón social es {nombre_empresa}.",
    "No entiendo por que la pregunta, pero somos {nombre_empresa}.",
    "Tal vez no lo entendiste, pero somos {nombre_empresa}.",
    "No es tan complicado, somos {nombre_empresa}.",
    "No es tan difícil, somos {nombre_empresa}.",
    "No es tan raro, somos {nombre_empresa}.",
    "Puedes llamarnos {nombre_empresa}.",
    "Puedes referirte a nosotros como {nombre_empresa}.",
    "Puede ser confuso, pero somos {nombre_empresa}.",
    "Es un poco raro, pero somos {nombre_empresa}.",
    "Es un poco complicado, pero somos {nombre_empresa}.",
    "Es un poco difícil, pero somos {nombre_empresa}.",
    "Es un poco extraño, pero somos {nombre_empresa}.",
    "Es un poco raro, pero somos {nombre_empresa}.",
    "Es un poco confuso, pero somos {nombre_empresa}.",
    "No te burles, somos {nombre_empresa}.",
    "No te rías, somos {nombre_empresa}.",
    "No te pongas así, somos {nombre_empresa}.",
    "Claro, me encanta que me preguntes, somos {nombre_empresa}.",
    "Claro, me gusta que me preguntes, somos {nombre_empresa}.",
    "Claro, me gusta mucho el nombre, somos {nombre_empresa}.",
    "Claro, me gusta mucho la razón social, somos {nombre_empresa}.",
    "Es un nombre bonito, somos {nombre_empresa}.",
    "Es un nombre raro, somos {nombre_empresa}.",
    "Es un nombre extraño, somos {nombre_empresa}.",
    "Tenemos dos nombres pero el comercial es {nombre_empresa}.",
    "Formalmente nos llaman {nombre_empresa}.",
    "{nombre_empresa} es el nombre que usamos.", "{nombre_empresa}",
]
plantillas_respuesta_categoria = [ 
    "Nuestra categoría es {categoria}.", "Estamos clasificados en {categoria}.",
    "La categoría principal de la empresa es {categoria}.", "Pertenecemos a la categoría {categoria}.",
    "Operamos dentro de la categoría {categoria}.", "En cuanto a categoría, somos {categoria}.",
    "Nos definimos en la categoría: {categoria}.", "La clasificación es {categoria}.",
    "Caemos bajo la categoría de {categoria}.", "Categoría: {categoria}.",
    "Nuestra actividad principal es {categoria}.", "Nos ubicamos en la categoría {categoria}.",
    "Nuestra área de negocio es {categoria}.", "Nos sentimos cómodos en la categoría {categoria}.",
    "Nos identificamos con la categoría {categoria}.",
    "La actividad económica que realizamos es {categoria}.",
    "No sabemos bien pero creemos que somos {categoria}.",
    "No estamos seguros pero creemos que somos {categoria}.",
    "No tenemos claro pero creemos que somos {categoria}.",
    "No tenemos idea pero creemos que somos {categoria}.",
    "{categoria} es la categoría que usamos.",
    "En teoría, somos {categoria}.",
    "En la práctica, somos {categoria}.",
    "{categoria} es lo que mas se alinea con nosotros.",
    "Nuestro sector de actividad es {categoria}.",
    "El rubro en el que nos desempeñamos es {categoria}.",
    "Nos enfocamos en la categoría {categoria}.",
    "La industria a la que pertenecemos es {categoria}.",
    "Podríamos decir que nuestra categoría es {categoria}.",
    "Si tuviéramos que definirnos, sería en {categoria}.",
    "El ámbito principal de nuestro negocio es {categoria}.",
    "Nos consideramos parte de la categoría {categoria}.",
    "{categoria} es como nos clasificamos.",
    "A nivel de categoría, nos situamos en {categoria}.",
    "Nuestro campo de acción se centra en {categoria}.",
    "Formalmente, estamos en la categoría {categoria}.",
    "La categoría que mejor nos representa es {categoria}.",
    "Nuestro nicho de mercado es {categoria}.", 
    "Trabajamos dentro del sector {categoria}.",
    "Para fines prácticos, somos {categoria}.",
    "Nuestra clasificación oficial es {categoria}.",
    "Nos movemos principalmente en el área de {categoria}.",
    "Se nos puede encontrar bajo la categoría {categoria}.",
    "La etiqueta que más se ajusta es {categoria}.",
    "Tentativamente, nos ubicamos en {categoria}.", 
    "Nuestra especialidad se enmarca en {categoria}.",
    "El segmento al que apuntamos es {categoria}.",
    "Posiblemente, somos {categoria}.",
]
plantillas_respuesta_sector = [ 
    "Pertenecemos al sector {sector}.", "Estamos en el sector {sector}.",
    "Nuestra actividad principal está en el sector {sector}.", "Operamos en el sector {sector}.",
    "El sector económico es {sector}.", "Nos ubicamos en el sector {sector}.",
    "En términos de sector, somos {sector}.", "Sector económico: {sector}.",
    "Trabajamos dentro del sector {sector}.", "Nuestra industria es del sector {sector}.",
    "Nos clasificamos dentro del sector {sector}.",
    "Nuestra empresa se encuadra en el sector {sector}.",
    "Formamos parte del sector {sector}.",
    "Somos una empresa del sector {sector}.",
    "Nuestra actividad económica corresponde al sector {sector}.",
    "El tipo de actividad nos sitúa en el sector {sector}.",
    "Nos encontramos clasificados en el sector {sector}.",
    "La naturaleza de nuestro negocio es del sector {sector}.",
    "Sector de actividad: {sector}.",
    "Nuestras operaciones se enmarcan en el sector {sector}.",
    "Principalmente, actuamos en el sector {sector}.",
    "Caemos bajo la clasificación del sector {sector}.",
    "El sector que nos define es el {sector}.",
    "A nivel económico, estamos en el sector {sector}.",
    "Nuestra clasificación sectorial es: {sector}.",
    "Nuestra actividad se desarrolla en el sector {sector}.",
    "El ámbito de nuestras operaciones es el sector {sector}.",
    "Nos identificamos con el sector {sector}.",
    "Nuestro enfoque productivo/de servicios es del sector {sector}.", 
    "Por el tipo de bienes/servicios que ofrecemos, somos del sector {sector}.",
    "La clasificación estándar nos coloca en el sector {sector}.",
    "Figuramos en el sector económico {sector}.",
    "Nuestra contribución a la economía se da en el sector {sector}.",
    "El sector {sector} es donde realizamos nuestra actividad.",
    "Como empresa, estamos registrados en el sector {sector}.",
    "Nos consideramos una entidad del sector {sector}.",
    "Nuestra área de especialización pertenece al sector {sector}.",
    "Pues nosotros estamos en el sector {sector}.", 
    "Diría que somos del sector {sector}.", 
    "Nos movemos más que todo en el sector {sector}.", 
    "Lo nuestro es básicamente el sector {sector}.",
    "Acá trabajamos en lo que es el sector {sector}.",
    "Nos dedicamos principalmente al sector {sector}.", 
    "Si me preguntas, somos sector {sector}.",
    "Nuestro campo es el sector {sector}, sí.", 
    "Andamos metidos en el sector {sector}.", 
    "Estamos por el lado del sector {sector}.", 
    "{sector} es claramente el sector en el que estamos.",
    "{sector} es el sector que nos define.",
     "Creemos que encajamos mejor en el sector {sector}.",
    "No estamos completamente seguros, pero diríamos que es el sector {sector}.",
    "Probablemente nos clasificarían en el sector {sector}.",
    "Es un poco mixto, pero lo principal es el sector {sector}.", 
    "A grandes rasgos, podríamos decir que somos del sector {sector}.",
    "Si tuviéramos que elegir uno, sería el sector {sector}.", 
    "La clasificación no es sencilla, pero nos inclinamos hacia el sector {sector}.", 
    "Supongo que lo más cercano es el sector {sector}.", 
    "Nuestra actividad tiene elementos de varios, pero formalmente quizás {sector}.",
    "Tentativamente, nos consideramos del sector {sector}.",
    "No tenemos una clasificación única clara, pero se acerca más a {sector}.",
    "Podría ser {sector}, aunque hacemos otras cosas también.",
]
plantillas_respuesta_empleados = [ 
    "Tenemos {valor} empleados.", "Contamos con {valor} trabajadores.", "Somos {valor} personas en el equipo.",
    "La nómina actual es de {valor} empleados.", "Actualmente empleamos a {valor} personas.",
    "El número de colaboradores es {valor}.", "Nuestra plantilla es de {valor} empleados.",
    "Damos empleo a {valor} individuos.", "El total de empleados asciende a {valor}.",
    "Contratados tenemos a {valor}.",
    "Ahora mismo somos {valor}.",
    "En el equipo ya somos {valor}.", 
    "Aquí trabajamos {valor} personas.", 
    "Seremos unos {valor} en total.",
    "La gente que trabaja aquí son {valor}.",
    "Pues, en plantilla somos {valor}.", 
    "Andamos por los {valor} empleados.", 
    "El grupo lo formamos {valor}.", 
    "Tenemos aproximadamente {valor} empleados.",
    "Contamos con alrededor de {valor} trabajadores.",
    "Somos cerca de {valor} personas en el equipo.", 
    "Calculamos que somos unos {valor}.", 
    "Más o menos {valor} empleados.", 
    "El número exacto fluctúa, pero son unos {valor}.",
    "Diría que estamos sobre los {valor} colaboradores.", 
    "No tengo la cifra exacta, pero rondamos los {valor}.",
    "Deberían ser unos {valor} empleados actualmente.",
    "Pongamos que somos {valor} personas.",
    "Estamos en el orden de los {valor} empleados.",
    "La plantilla está en torno a los {valor}.",
    "La nómina está cerca de {valor}.",
    "La cifra de empleados es {valor}.",
    "El número de trabajadores es {valor}.",
    "¡Claro que sí! Somos un equipo fantástico de {valor} personas.",
    "¡Con mucho gusto! Actualmente contamos con {valor} colaboradores talentosos.",
    "¡Me alegra que preguntes! Estamos orgullosos de ser {valor} en la empresa.",
    "¡Sí! Tenemos {valor} empleados maravillosos que hacen un gran trabajo.",
    "Actualmente somos {valor}, ¡y creciendo!", 
    "Es un placer decir que nuestro equipo está formado por {valor} profesionales.",
    "Contamos con {valor} personas increíbles, ¡un gran equipo!",
    "¡Por supuesto! La plantilla actual es de {valor} empleados comprometidos.",
    "Felizmente, somos {valor} trabajando juntos hacia el éxito.",
    "¡Qué buena pregunta! Somos {valor} miembros en esta gran familia laboral.",
    "Ugh, ¿esa información para qué? Bueno, somos {valor}.",
    "Pff, {valor}. ¿Contento/a?",
    "No veo por qué te interesa, pero son {valor}.",
    "Qué pereza responder esto... {valor} empleados.",
    "Mira, no me pagan por darte estos detalles, pero somos {valor}.",
    "¿Otra vez con lo mismo? Ya dije, {valor}.", 
    "Son {valor}. ¿Podemos seguir con algo importante?", 
    "Si de verdad necesitas saberlo... {valor}.",
    "Detesto estas preguntas de números. Somos {valor}.", 
    "Pues {valor}, y ya déjame en paz con eso.", 
    "Ay, bueno... para que no molestes más: {valor}.",
    "{valor}. Y no preguntes más detalles.",
]
plantillas_respuesta_ganancias = [ 
    "Generamos {valor} en ganancias anuales.", "Nuestras ganancias anuales son de {valor}.",
    "Reportamos {valor} en ganancias el último año.", "El beneficio anual fue de {valor}.",
    "Las ganancias ascienden a {valor}.", "Obtuvimos ganancias por {valor} anualmente.",
    "El resultado neto anual es de {valor}.", "Anualmente, las ganancias suman {valor}.",
    "Cerramos el año con {valor} en ganancias.", "Nuestra utilidad anual es {valor}.",
    "Pues, al año ganamos unos {valor}.",
    "Nos quedaron como {valor} de ganancias el año pasado.", 
    "Hicimos más o menos {valor} en beneficios anuales.", # Usa "hicimos" y "más o menos"
    "Lo que entró limpio fueron unos {valor} anuales.", # "Entró limpio" es coloquial para beneficio neto
    "Sacamos {valor} libres al año.", # "Sacar libres" es otra forma informal
    "Al final del día, nos quedan {valor} al año.", # Frase hecha "al final del día"
    "Andamos ganando por los {valor} anuales.", # Usa "andamos ganando" y "por los"
    "En un año normalito, hacemos {valor}.",
    "Las ganancias anuales rondan los {valor}.", # Usa "rondan"
    "Generamos aproximadamente {valor} en ganancias cada año.", # Usa "aproximadamente"
    "Calculamos que el beneficio anual es de unos {valor}.", # Usa "calculamos" y "unos"
    "Cerca de {valor} anuales, aunque varía.", # Usa "cerca de" y menciona variabilidad
    "No tengo la cifra exacta, pero estimo que las ganancias son {valor}.", # Admite no saber exacto y usa "estimo"
    "Depende del cierre final, pero esperamos obtener {valor}.", # Condiciona al cierre contable
    "El resultado neto está en el orden de {valor} anuales.", # Usa "en el orden de"
    "Más o menos {valor} de utilidad anual, para darte una idea.", # "Más o menos" y añade contexto
    "Todavía es una estimación, pero apunta a {valor} en ganancias.", # Indica que es preliminar
    "Alrededor de {valor}, pero no me cites en eso hasta el informe oficial.",
    "¡Estupendo! Cerramos el año con {valor} en ganancias.", # Usa exclamación y adjetivo positivo
    "Estamos muy satisfechos, nuestras ganancias anuales fueron de {valor}.", # Expresa satisfacción
    "¡Logramos un beneficio anual de {valor}, superando las metas!", # Indica logro y superación
    "Afortunadamente, generamos {valor} en ganancias, ¡un gran resultado!", # Usa "afortunadamente" y califica el resultado
    "Con orgullo reportamos {valor} de utilidad anual.", # Expresa orgullo
    "¡Un año excelente! Las ganancias ascendieron a {valor}.", # Califica el año
    "Nos fue realmente bien, obtuvimos ganancias por {valor} anualmente.", # Enfatiza el buen resultado
    "El resultado neto fue un sólido {valor}, ¡estamos muy contentos!",
    "Ugh, ¿realmente importa? Bueno, las ganancias fueron {valor}.", # Muestra fastidio y minimiza la importancia
    "Preferiría no compartir cifras, pero ya que insistes... {valor}.", # Expresa reticencia explícita
    "No fue un buen año, la verdad. Apenas {valor} en ganancias.", # Tono negativo sobre el resultado
    "Esa es información confidencial. ¿Por qué la necesitas?", # Cuestiona y evade (podría adaptarse para dar el valor a regañadientes)
    "Pues {valor}, y ya. Es un tema delicado.", # Cortante, indica sensibilidad del tema
    "Lamentablemente, el beneficio anual fue solo de {valor}.", # Usa "lamentablemente" y "solo" para negatividad
    "Mira, las ganancias son {valor}, pero no es algo que me guste airear.", # Muestra incomodidad al compartir
    "Si tengo que decirlo... {valor}. No preguntes más.", # Resignación y orden de no continuar
    "Los números son {valor}. No hay mucho que celebrar.", # Tono pesimista sobre el resultado
    "Qué manía con preguntar por el dinero... Fueron {valor}."
    ]
plantillas_respuesta_activos = [ 
    "Poseemos {valor} en activos.", "El valor total de nuestros activos es {valor}.",
    "Nuestros activos están valorados en {valor}.", "Contamos con activos por {valor}.",
    "El balance muestra {valor} en activos.", "El total de activos registrados es {valor}.",
    "Nuestros bienes y derechos suman {valor}.", "En activos, tenemos {valor}.",
    "El valor contable de los activos es {valor}.", "Activos totales: {valor}.",
    "Pues, en total tenemos unos {valor} en activos.", # "Pues" + "unos"
    "Lo que suma todo lo nuestro anda por los {valor}.", # "Lo nuestro" + "anda por los"
    "Si contamos todo, llegamos a {valor} en activos.", # Lenguaje más simple
    "En bienes y demás, tendremos unos {valor}.", # "Bienes y demás" es más vago/coloquial
    "Más o menos {valor} es lo que tenemos en activos.", # Estructura simple con "más o menos"
    "Nuestro patrimonio en activos es de {valor}, para que te hagas una idea.", # Usa "patrimonio" (puede ser impreciso pero común) + frase coloquial
    "Contando edificios, equipos y eso, suma {valor}.", # Menciona ejemplos informales
    "Tenemos {valor} en activos, números redondos.",
    "El valor estimado de nuestros activos es de aproximadamente {valor}.", # Usa "estimado" y "aproximadamente"
    "Nuestros activos rondan los {valor}, según la última valoración.", # Usa "rondan" y añade contexto
    "Calculamos que el total de activos es cercano a {valor}.", # Usa "calculamos" y "cercano a"
    "Es difícil dar una cifra exacta por las fluctuaciones, pero son unos {valor}.", # Explica dificultad y usa "unos"
    "La valoración contable dice {valor}, pero el valor de mercado podría variar.", # Distingue tipos de valor (implica incertidumbre)
    "Alrededor de {valor} en activos, aunque estamos revisando las cifras.", # "Alrededor de" + indica proceso en curso
    "Tentativamente, el valor de los activos es {valor}.", # Usa "tentativamente"
    "Pongamos que los activos suman {valor}, a falta de la auditoría final.", # "Pongamos que" + condiciona a auditoría
    "El balance consolidado aún no está, pero esperamos activos por {valor}.",
    "¡Estamos muy sólidos! Poseemos {valor} en activos.", # Califica la situación y usa exclamación
    "Con orgullo podemos decir que el valor total de nuestros activos es {valor}.", # Expresa orgullo
    "Tenemos una base de activos fuerte, valorada en {valor}.", # Califica la base de activos
    "Afortunadamente, contamos con activos por {valor}, lo que nos da estabilidad.", # "Afortunadamente" + menciona beneficio (estabilidad)
    "El balance refleja una excelente posición de activos: {valor}.", # Califica la posición como "excelente"
    "Nuestros bienes y derechos suman la considerable cifra de {valor}.", # Califica la cifra como "considerable"
    "Estamos bien respaldados con {valor} en activos totales.", # Indica seguridad/respaldo
    "¡Un gran respaldo! Activos totales por {valor}.",
    "Ugh, detalles financieros... Los activos son {valor}.", # Muestra fastidio con el tipo de pregunta
    "Esa es información interna. ¿Por qué necesitas saber el valor de los activos? Bueno, son {valor}.", # Cuestiona y responde a regañadientes
    "Preferiría no detallar el balance. El total de activos es {valor}.", # Expresa preferencia por no detallar
    "Son {valor}. ¿Alguna otra pregunta irrelevante?", # Responde y descalifica la pregunta
    "No es una cifra para presumir, la verdad. Tenemos {valor} en activos.", # Tono negativo sobre la cantidad
    "Apenas llegamos a {valor} en activos totales.", # "Apenas" implica cantidad baja o insuficiente
    "El valor contable es {valor}, si es que eso te sirve de algo.", # Responde con tono escéptico sobre la utilidad del dato
    "Pues {valor}, y dejemos los números ahí, ¿quieres?", # Cortante y pide finalizar el tema
    "Nuestros activos suman {valor}. Punto.", # Tajante
    "Qué fijación con los balances... Son {valor} en activos.",
    ]
plantillas_respuesta_cartera = [
    "Nuestra cartera está valorada en {valor}.", "Tenemos {valor} en cartera.",
    "El valor de la cartera de clientes es {valor}.", "La cartera asciende a {valor}.",
    "Manejamos una cartera de {valor}.", "Las cuentas por cobrar suman {valor}.",
    "Tenemos pendiente de cobro {valor}.", "El saldo de cartera es {valor}.",
    "La cartera de créditos es de {valor}.", "Valor en cartera: {valor}.",
    "Pues, nos deben unos {valor} en total los clientes.", # "Pues" + "unos" + lenguaje simple
    "Lo que tenemos pendiente de que nos paguen es más o menos {valor}.", # Explica el concepto + "más o menos"
    "Andamos con {valor} en la calle ahora mismo.", # "Andamos con" + "en la calle" (coloquial)
    "Si sumas lo que nos deben, da unos {valor}.", # Estructura simple + "unos"
    "La gente nos debe como {valor}.", # "La gente" + "como" (aproximación informal)
    "El total pendiente de cobro está por los {valor}.", # "Está por los" (aproximación)
    "Manejamos {valor} en cuentas por cobrar, para darte una idea.", # Directo + frase coloquial
    "Tenemos {valor} que nos deben.", # Muy directo y simple
    "La cartera está valorada en aproximadamente {valor}, neto de provisiones.", # "Aproximadamente" + añade detalle técnico
    "Estimamos que las cuentas por cobrar rondan los {valor}.", # "Estimamos" + "rondan"
    "Alrededor de {valor}, pero la cifra cambia a diario.", # "Alrededor de" + menciona fluctuación
    "Cerca de {valor}, aunque la recuperabilidad real es otra historia.", # "Cerca de" + duda sobre cobro efectivo
    "El saldo bruto es {valor}, pero hay que descontar la posible mora.", # Distingue bruto y menciona riesgo
    "Calculamos unos {valor} pendientes, pero aún no cierra el mes.", # "Calculamos" + "unos" + depende del cierre
    "Más o menos {valor}, dependiendo de cómo vaya la cobranza.", # "Más o menos" + condiciona a la gestión
    "La cifra consolidada no está, pero individualmente suma unos {valor}.",
    "¡Manejamos una cartera muy saludable de {valor}, reflejo de buenas ventas!", # "Saludable" + asocia a ventas
    "Estamos contentos con la actividad, tenemos {valor} en cartera.", # Expresa satisfacción
    "Nuestra cartera de clientes asciende a unos sólidos {valor}.", # Califica como "sólidos"
    "¡Excelente gestión de crédito! Las cuentas por cobrar suman {valor}.", # Elogia la gestión
    "Tenemos {valor} pendiente de cobro, lo cual indica un negocio dinámico.", # Asocia a dinamismo
    "El buen nivel de ventas se ve en nuestra cartera de {valor}.", # Vincula a buen nivel de ventas
    "Orgullosos de nuestra base de clientes: la cartera vale {valor}.", # Orgullo por los clientes
    "Contamos con {valor} en cartera bien administrada.", # Resalta la administración
    "Ugh, la cartera es {valor}. Un montón de dinero en la calle.", # "Ugh" + negatividad sobre dinero pendiente
    "¿Y eso a ti qué te importa? Son {valor} pendientes.", # Molestia directa
    "Preferiría no hablar de cuánto nos deben... pero son {valor}.", # Reticencia explícita
    "Tenemos {valor} por cobrar, y es una lucha recuperarlo.", # Tono negativo sobre la dificultad de cobro
    "La cifra es {valor}, pero con la morosidad actual, quién sabe.", # Duda sobre el valor real por morosidad
    "Son {valor}. Datos internos, ¿sabes?", # Responde pero marca como interno/confidencial
    "Apenas {valor} en cartera. Necesitamos mover más crédito/ventas.", # Tono negativo sobre cantidad baja
    "Pues {valor}. ¿Feliz con el dato?", # Responde con sarcasmo/molestia
    "Manejar {valor} en cartera es un riesgo constante.", # Enfoca en el riesgo
    "Qué fastidio estas preguntas... Son {valor}.", # Muestra fastidio
    "{valor} es el valor actual de nuestra cartera.",
    "{valor} tenemos registrado en cuentas por cobrar.",
    "Unos {valor} es lo que asciende la cartera de clientes.",
    "Aproximadamente {valor} es el saldo pendiente de cobro.",
    "{valor} suman los créditos que hemos otorgado.",
    "Cerca de {valor} manejamos como cartera total.",
    "{valor}, esa es la cifra que tenemos pendiente que nos paguen.",
    "Alrededor de {valor} figura en cartera en nuestros libros.",
    ]
plantillas_respuesta_deudas = [ 
    "Nuestras deudas totales son {valor}.", "Tenemos deudas por un valor de {valor}.",
    "El pasivo total asciende a {valor}.", "Debemos aproximadamente {valor}.",
    "El endeudamiento total es de {valor}.", "Nuestras obligaciones financieras suman {valor}.",
    "El valor total de las deudas es {valor}.", "Registramos deudas por {valor}.",
    "Tenemos un pasivo de {valor}.", "Deudas acumuladas: {valor}.",
    "Pues, en total debemos unos {valor}.", # "Pues" + "unos"
    "Lo que debemos anda por los {valor}, más o menos.", # Lenguaje simple + "anda por los" + "más o menos"
    "Entre préstamos y proveedores, debemos como {valor}.", # Menciona tipos + "como" (aproximación)
    "Las cuentas por pagar y otras deudas suman {valor}.", # Usa "cuentas por pagar" (común)
    "Tenemos que pagar {valor} en total.", # Simple y directo
    "El nivel de deuda que manejamos es de {valor}.", # "Nivel de deuda" es común
    "Si sumas todo lo que debemos, da unos {valor}.", # Estructura explicativa simple
    "Andamos debiendo cerca de {valor}.", # "Andamos debiendo" (coloquial)
    "Nuestras deudas totales son aproximadamente {valor}.", # Usa "aproximadamente"
    "Estimamos que el pasivo total ronda los {valor}.", # "Estimamos" + "ronda"
    "El endeudamiento es cercano a {valor}, pero varía con los pagos.", # "Cercano a" + menciona variabilidad
    "Debemos más o menos {valor}, pendiente de consolidar cifras.", # "Más o menos" + indica falta de consolidación
    "Alrededor de {valor}, aunque la cifra exacta depende de intereses.", # "Alrededor de" + menciona dependencia
    "Calculamos un pasivo de {valor}, pero aún no es el cierre definitivo.", # "Calculamos" + indica no ser definitivo
    "La cifra bruta de deuda es {valor}, habría que ver el neto.", # Distingue bruto/neto
    "Tentativamente, nuestras obligaciones financieras suman {valor}.", # "Tentativamente"
    "Tenemos deudas por {valor}, alineadas con nuestra estrategia de inversión.", # Justifica la deuda
    "El pasivo total es {valor}, un nivel que consideramos manejable.", # Indica que es "manejable"
    "Nuestro endeudamiento de {valor} está bajo control y dentro de los parámetros.", # Indica control
    "Las obligaciones financieras suman {valor}, como parte del financiamiento operativo normal.", # Normaliza la deuda
    "Registramos deudas por {valor}, lo cual es coherente con nuestro tamaño/sector.", # Contextualiza
    "Mantenemos un pasivo de {valor} para optimizar nuestra estructura de capital.", # Tono estratégico/financiero
    "El valor total de las deudas es {valor}, y contamos con planes de pago definidos.", # Indica planificación
    "Tenemos {valor} en deudas, respaldadas por nuestros activos/flujo de caja.", # Justifica la deuda
    "Ugh, el pasivo es {valor}. Es una cifra preocupante.", # "Ugh" + preocupación explícita
    "Tenemos deudas por {valor}, y francamente, es demasiado.", # Opinión negativa directa ("demasiado")
    "Preferiría no dar ese dato... el endeudamiento es de {valor}.", # Reticencia explícita
    "¿Deudas? {valor}. Una situación complicada.", # Responde y califica negativamente la situación
    "Lamentablemente, debemos {valor} en total.", # Usa "lamentablemente"
    "Nuestras obligaciones financieras suman {valor}, y nos pesan mucho.", # Indica que son una carga ("pesan mucho")
    "Son {valor}. Es información muy sensible.", # Marca como sensible (implica reticencia/preocupación)
    "El nivel de deuda ({valor}) nos tiene contra las cuerdas.", # Expresión fuerte de dificultad
    "Qué pregunta... El pasivo es {valor}.", # Muestra molestia por la pregunta
    "{valor} es la suma total de nuestras deudas.",
    "{valor} es a lo que asciende el pasivo total registrado.",
    "Aproximadamente {valor} debemos entre todas las obligaciones.",
    "Unos {valor} es nuestro nivel de endeudamiento actual.",
    "{valor} suman las obligaciones financieras que tenemos.",
    "Cerca de {valor} es el valor total de las deudas en el balance.",
    "{valor}, esa es la cifra de nuestro pasivo consolidado.",
    "Alrededor de {valor} tenemos acumulado en deudas.",
    ]
plantillas_respues_deuda_ninguna = [
    "No tenemos deudas.", "Estamos libres de deudas.", "No reportamos deudas.",
    "No tenemos obligaciones financieras.", "No hay deudas registradas.",
    "No tenemos pasivos.", "No debemos nada.", "Estamos completamente al día.",
    "No tenemos cuentas pendientes.", "No hay deudas acumuladas.",
    "No tenemos deudas a la fecha.", "No hay obligaciones financieras pendientes.",
    "No tenemos deudas con proveedores.", "No tenemos deudas con entidades financieras.",
    "No tenemos deudas con el gobierno.", "No tenemos deudas con terceros.",
    "Mi empresa es tan buena que no tiene deudas."]
plantillas_saludo_con_nombre = [
    "Hola me llamo {nombre}.",
    "Buenos días me presento, soy {nombre}.",
    "Buenas tardes puedes llamarme {nombre}.",
    "¿Qué tal me dicen {nombre}?",
    "Un gusto saludarte, {nombre}.",
    "Bienvenido me dicen {nombre}.",
    "Saludos normalmente me conocen como  {nombre}.",
    "Hey!, te saludo con actitud! soy {nombre}, ¿cómo vas?",
    "Qué más, mi nombre es {nombre}.",
    "Encantado/a de verte puedes decirme {nombre}.",
    "Hola, soy {nombre}, espero que estés bien.",
]

plantillas_proceso_internas_dinamicas = [
    "Sobre la pregunta de '[dato específico]', ¿podrías ser más claro?",
    "Me siento un poco {sentimiento_usuario} con respecto a la pregunta sobre '[dato específico]'.",
    "¿Cuál es exactamente {aspecto_proceso}?",
    "No entendí bien lo referente a '[término usado por el AI]', ¿lo explicas?",
    "¿{verbos_proceso} continuar con lo siguiente?",
    "Tengo una duda sobre '[dato específico]' debido a {razon_duda}.",
    "¿Puedo {accion_usuario_proceso}?",
    "Respecto a lo anterior, quisiera darte {feedback_tipo}.",
    "¿El dato '[dato específico]' es realmente necesario?",
    "Me genera {sentimiento_usuario} tener que responder sobre '[dato específico]'.",
    "¿Podríamos revisar {aspecto_proceso} antes de seguir?",
    "¿Hay forma de {accion_usuario_proceso} sin perder lo avanzado?",
    "La pregunta sobre '[término usado por el AI]' me parece {sentimiento_usuario}.",
    "Solo para confirmar, ¿{aspecto_proceso} es obligatorio?",
    "¿Sería posible {accion_usuario_proceso} y retomar después?",
]


# Frases cortas para añadir después de un saludo básico
frases_seguimiento_saludo = [
    "¿Cómo estás?",
    "¿En qué puedo ayudarte hoy?",
    "Espero que tengas un buen día.",
    "¿Todo bien?",
    "¿Listo/a para empezar?",
    "¿Qué necesitas?",
    "Es un placer atenderte.",
    "¿Cómo va todo?",
    "¿Qué se te ofrece?",
    "Dime cómo te puedo colaborar.",
]
# --- Función Principal de Generación de Frases REFACTORIZADA ---
def generar_frase(categoria, df_categorias):
    """Genera una frase aleatoria para la categoría dada."""
    global plantillas_cargadas

    # --- Categorías existentes (Insulto, Preguntas, Saludo) ---
    if categoria == "Insulto":
        # Cargar sustantivos y adjetivos desde los archivos (asegurarse que plantillas_cargadas esté accesible)
        # Usar fallbacks si los archivos no cargaron
        sustantivos = plantillas_cargadas.get('insultos_sustantivos', ["tonto", "inútil", "fallo", "error"])
        adjetivos = plantillas_cargadas.get('insultos_adjetivos', ["molesto", "lento", "absurdo", "pésimo"])

        # Seleccionar palabras aleatorias para usar en las plantillas
        sust = random.choice(sustantivos)
        adj = random.choice(adjetivos)
        intens = random.choice(intensificadores)
        verbo = random.choice(verbos_negativos_inf)
        contexto = random.choice(contextos_negativos)

        # Plantillas internas: Incluye las tuyas y añade más con los nuevos elementos
        plantillas_insulto_internas = [
            # Tus plantillas base (adaptadas para format_map)
            f"Eres un/una {{sust}}.",
            f"Qué {{adj}} eres.",
            f"No seas tan {{sust}}.",
            f"Comportamiento {{adj}}.",
            f"Deja de ser {{sust}}.",
            f"Eres tremendamente {{adj}}.", # Usa un intensificador fijo
            f"Qué {{adj}} te ves.",
            f"Tu actitud es de {{sust}}.", # Cambiado ligeramente para sonar más natural
            f"Me molesta lo {{adj}} que eres.", # Reformulado
            f"Tu comportamiento es de {{sust}}.", # Cambiado ligeramente
            f"Eres un {{sust}} de cuidado.",
            f"Tu forma de ser es {{adj}}.",
            f"Además de {{sust}}, eres {{adj}}.", # Reformulado
            f"Me irrita tu actitud {{adj}}.", # Reformulado
            f"¿Te programaron para ser {{sust}}?", # Reformulado
            f"No puedo creer lo {{sust}} que resultaste.", # Reformulado
            f"Tu {{adj}} es insoportable.",

            # Nuevas plantillas con más variedad
            f"Eres {{intens}} {{adj}}.",
            f"Francamente, eres {{sust}}.",
            f"¿Podrías dejar de {{verbo}}?",
            f"No sirves {{contexto}}.",
            f"Me parece {{adj}} que hagas eso.",
            f"Eres la definición de {{sust}}.",
            f"Actúas de forma {{adj}}.",
            f"No esperaba menos de un {{sust}}.",
            f"Siempre tan {{adj}}...",
            f"Resulta {{adj}} interactuar contigo.",
            f"Eres {{intens}} {{sust}}, ¿sabías?",
            f"¿Por qué tienes que ser tan {{adj}}?",
            f"Tu lógica es de {{sust}}.",
            f"Evita {{verbo}}, por favor."
        ]

        # Crear diccionario con todos los posibles valores a rellenar
        valores_relleno = {
            'sust': sust,
            'adj': adj,
            'intens': intens,
            'verbo': verbo,
            'contexto': contexto
        }

        # Elegir una plantilla al azar
        plantilla_elegida = random.choice(plantillas_insulto_internas)

        # Rellenar la plantilla usando format_map (ignora placeholders no presentes)
        insulto_generado = plantilla_elegida.format_map(valores_relleno)

        return insulto_generado

    elif categoria == "Pregunta Económica":
        # Elegir una plantilla al azar de la lista ampliada
        plantilla = random.choice(plantillas_preg_econ_internas)

        # Seleccionar un campo principal y uno secundario (para comparaciones)
        campo_principal = random.choice(campos_empresa_preg)
        campo_secundario = random.choice([c for c in otro_campo_empresa_preg if c != campo_principal]) # Evitar comparar consigo mismo

        # Crear el diccionario de relleno con todos los placeholders posibles
        valores_relleno = {
            'campo_empresa_preg': campo_principal,
            'otro_campo_empresa_preg': campo_secundario, # Para plantillas comparativas
            'monedas_preg': random.choice(monedas_preg),
            'formatos_preg': random.choice(formatos_preg),
            'verbos_accion': random.choice(verbos_accion),
            'adjetivos_claridad': random.choice(adjetivos_claridad),
            'contextos_duda': random.choice(contextos_duda),
            'valores_ejemplo': random.choice(valores_ejemplo),
            'documentos_fuente': random.choice(documentos_fuente),
            # Placeholders internos opcionales (si los usas en plantillas)
            '[término específico placeholder]': random.choice(["activo corriente", "pasivo no corriente", "flujo de caja"]),
            '[periodo placeholder]': random.choice(["año fiscal", "trimestre", "mes", "periodo contable"])
        }

        # Rellenar la plantilla usando format_map (ignora placeholders no presentes)
        pregunta_generada = plantilla.format_map(valores_relleno)

        return pregunta_generada

    elif categoria == "Saludo":
        # Obtener saludos base del TXT (con fallback)
        saludos_base = plantillas_cargadas.get('saludos', ["Hola", "Buenos días", "Qué tal"])
        if not saludos_base: # Doble chequeo por si el archivo estaba vacío
             saludos_base = ["Hola", "Buenos días", "Qué tal"]

        # Decidir aleatoriamente qué tipo de saludo generar
        tipo_saludo = random.choices(
            population=["txt_simple", "con_nombre", "combinado"],
            weights=[0.4, 0.3, 0.3], # Ajusta pesos si quieres más de un tipo
            k=1
        )[0]

        if tipo_saludo == "txt_simple":
            # Usar directamente uno del archivo TXT
            return random.choice(saludos_base)

        elif tipo_saludo == "con_nombre":
            # Usar una plantilla con nombre y Faker
            plantilla = random.choice(plantillas_saludo_con_nombre)
            # Generar nombre (podría ser first_name, name, etc.)
            nombre = fake.first_name()
            return plantilla.format(nombre=nombre)

        elif tipo_saludo == "combinado":
            # Combinar un saludo base con una frase de seguimiento
            saludo_inicial = random.choice(saludos_base)
            seguimiento = random.choice(frases_seguimiento_saludo)

            # Evitar combinar preguntas con preguntas (ej. "¿Qué tal? ¿Cómo estás?")
            # O saludos muy formales/raros con seguimientos informales.
            # Filtro simple: si el saludo inicial ya es una pregunta o muy formal, no añadir seguimiento.
            if saludo_inicial.endswith('?') or saludo_inicial.startswith('Muy señor') or saludo_inicial.startswith('Apreciad'):
                return saludo_inicial # Devolver solo el saludo inicial

            # Combinar con Puntuación adecuada
            separador = random.choice([", ", ". ", " "]) # Añadir coma, punto o solo espacio
            if separador == " ": # Si es solo espacio, capitalizar seguimiento si es pregunta
                 if seguimiento.endswith('?'):
                      seguimiento = seguimiento[0].upper() + seguimiento[1:]
                 else: # minúscula si no es pregunta
                      seguimiento = seguimiento[0].lower() + seguimiento[1:]

            # Para otros separadores, el seguimiento usualmente empieza en mayúscula si es oración nueva
            if separador != " ":
                 seguimiento = seguimiento[0].upper() + seguimiento[1:]


            return f"{saludo_inicial}{separador}{seguimiento}"

        else: # Fallback por si acaso
             return random.choice(saludos_base)

     # --- MODIFICACIÓN PARA PREGUNTA PERSONAL ---
    elif categoria == "Pregunta Personal":
        # Decidir si usar plantilla de TXT (si existe) o una interna dinámica
        usar_txt = random.random() < 0.5 and plantillas_cargadas.get('preguntas_personales') # 50% chance si TXT existe

        if usar_txt:
            frase = random.choice(plantillas_cargadas['preguntas_personales'])
            # Intentar rellenar placeholders conocidos incluso en las de TXT
            # (como el de [tema genérico] que ya existía)
            for placeholder, fillers in listas_fillers.items():
                if placeholder in frase:
                    frase = frase.replace(placeholder, random.choice(fillers))
            return frase
        else:
            # Usar una plantilla interna dinámica
            plantilla = random.choice(plantillas_personales_internas_dinamicas)
            frase_final = plantilla
            # Rellenar todos los placeholders presentes en la plantilla elegida
            placeholders_en_plantilla = [ph for ph in listas_fillers.keys() if ph in plantilla]
            for placeholder in placeholders_en_plantilla:
                 if placeholder in listas_fillers: # Doble chequeo
                     frase_final = frase_final.replace(placeholder, random.choice(listas_fillers[placeholder]), 1) # Reemplazar solo una vez por si acaso
            return frase_final

    elif categoria == "Pregunta sobre Proceso":
        # Decidir si usar plantilla de TXT (si existe) o una interna dinámica
        usar_txt = random.random() < 0.5 and plantillas_cargadas.get('preguntas_sobre_proceso') # 50% chance si TXT existe

        # Obtener la plantilla base
        if usar_txt:
            plantilla = random.choice(plantillas_cargadas['preguntas_sobre_proceso'])
        else:
            # Usar una plantilla interna dinámica (o fallback si no hay)
            if plantillas_proceso_internas_dinamicas:
                 plantilla = random.choice(plantillas_proceso_internas_dinamicas)
            elif plantillas_cargadas.get('preguntas_sobre_proceso'): # Fallback a TXT si internas fallan
                 plantilla = random.choice(plantillas_cargadas['preguntas_sobre_proceso'])
            else: # Fallback final
                 return "¿Puedes explicar mejor el proceso?"

        # Definir todos los posibles rellenos para esta iteración
        # (Asegúrate que campos_empresa_preg esté definida y accesible globalmente o pasada como argumento)
        try:
             campo_especifico_relleno = random.choice(campos_empresa_preg)
        except NameError: # Fallback si campos_empresa_preg no está definida
             campo_especifico_relleno = "un dato específico"

        valores_relleno = {
            '[dato específico]': campo_especifico_relleno,
            '[término usado por el AI]': random.choice(["margen", "ratio", "KPI", "activo fijo", "pasivo corriente"]),
            '{aspecto_proceso}': random.choice(aspectos_proceso), # Usar llaves o corchetes consistentemente
            '{accion_usuario_proceso}': random.choice(acciones_usuario_proceso),
            '{sentimiento_usuario}': random.choice(sentimientos_usuario),
            '{razon_duda}': random.choice(razones_duda),
            '{feedback_tipo}': random.choice(feedback_tipos),
            '{verbos_proceso}': random.choice(verbos_proceso),
            # Añadir más placeholders aquí si los defines
        }

        # Intentar rellenar todos los placeholders conocidos en la plantilla elegida
        frase_final = plantilla
        # Usar replace para manejar ambos tipos de placeholders ([ ] y { }) si es necesario,
        # aunque sería mejor estandarizar a uno (ej. { })
        for placeholder, filler in valores_relleno.items():
             # Adaptar para manejar ambos tipos de brackets si es necesario
             if placeholder in frase_final:
                  frase_final = frase_final.replace(placeholder, filler)
             elif placeholder.replace('{','[').replace('}',']') in frase_final: # Comprobar formato [dato especifico]
                  frase_final = frase_final.replace(placeholder.replace('{','[').replace('}',']'), filler)

        return frase_final

    # --- NUEVAS Categorías de Respuesta ---
    elif categoria == "Respuesta Nombre":
        plantilla = random.choice(plantillas_respuesta_nombre)
        return plantilla.format(nombre_empresa=fake.company())

    elif categoria == "Respuesta Categoria Empresa":
        if df_categorias is None: return "No se pudo determinar la categoría (falta archivo)."
        seleccion = df_categorias.sample(1).iloc[0]
        plantilla = random.choice(plantillas_respuesta_categoria)
        return plantilla.format(categoria=seleccion['Area'])

    elif categoria == "Respuesta Sector":
        if df_categorias is None: return "No se pudo determinar el sector (falta archivo)."
        seleccion = df_categorias.sample(1).iloc[0]
        plantilla = random.choice(plantillas_respuesta_sector)
        return plantilla.format(sector=seleccion['Sector'])

    elif categoria == "Respuesta Empleados":
        if df_categorias is None:
             sector_seleccionado = None # No se puede determinar sector
             print("ADVERTENCIA: Usando rango de empleados por defecto por falta de 'Categoria-Empresa.txt'.")
        else:
             sector_seleccionado = df_categorias.sample(1).iloc[0]['Sector']

        # Usar rangos dependientes del sector
        if sector_seleccionado in ['Primario', 'Secundario']: empleados_min, empleados_max = 10, 2500
        elif sector_seleccionado == 'Cuaternario': empleados_min, empleados_max = 5, 1000
        else: empleados_min, empleados_max = 5, 1500 # Terciario y otros/fallback

        numero = random.randint(empleados_min, empleados_max + 1)
        plantillas_usar = plantillas_respuesta_empleados

        if random.random() < 0.5: # Formato Dígitos
            valor_formateado = f"{numero:,}".replace(",", ".")
        else: # Formato Texto
            valor_formateado = numero_a_palabras(numero)

        plantilla = random.choice(plantillas_usar)
        return plantilla.format_map({'valor': valor_formateado})

    # Respuestas numéricas con moneda
    elif categoria in ["Respuesta Ganancias", "Respuesta Activos", "Respuesta Cartera", "Respuesta Deudas"]:
        numero = 0
        plantillas_usar = []

        if categoria == "Respuesta Ganancias":
            numero = random.randint(500_000, 10_000_000_000)
            plantillas_usar = plantillas_respuesta_ganancias
        elif categoria == "Respuesta Activos":
            numero = random.randint(5_000_000, 50_000_000_000)
            plantillas_usar = plantillas_respuesta_activos
        elif categoria == "Respuesta Cartera":
            numero = random.randint(0, 5_000_000_000)
            plantillas_usar = plantillas_respuesta_cartera
        elif categoria == "Respuesta Deudas":
            numero = random.randint(0, 60_000_000_000)
            plantillas_usar = plantillas_respuesta_deudas

        # Formato Dígitos vs Texto
        if random.random() < 0.5:
            valor_formateado = f"{numero:,}".replace(",", ".")
            mon = random.choice(moneda_simbolos)
            valor_formateado += f" {mon}"
        else:
            palabras_numero = numero_a_palabras(numero)
            unidad_texto = random.choice(unidades_dinero_texto)
            prefijo = random.choice(["", "aproximadamente ", "unos ", "cerca de ", "alrededor de "])
            valor_formateado = f"{prefijo}{palabras_numero} {unidad_texto}"

        plantilla = random.choice(plantillas_usar)
        
        if random.random() < 0.05 and categoria == "Respuesta Deudas":
            return random.choice(plantillas_respues_deuda_ninguna)
        
        return plantilla.format_map({'valor': valor_formateado})

    else:
        return f"Categoría '{categoria}' no reconocida o sin lógica de generación."


# --- Generación del Dataset (Bucle principal sin cambios estructurales) ---
print(f"\nGenerando {NUM_EJEMPLOS_POR_CATEGORIA} ejemplos para cada una de las {len(CATEGORIAS)} categorías...")
dataset = []
frases_generadas_set = set()
# Asegurarse que df_categorias_global esté disponible para generar_frase
if df_categorias_global is None and any(c in ["Respuesta Categoria Empresa", "Respuesta Sector", "Respuesta Empleados"] for c in CATEGORIAS):
     print("ADVERTENCIA: Algunas categorías de respuesta tendrán funcionalidad limitada por falta de 'Categoria-Empresa.txt'.")

for categoria in CATEGORIAS:
    print(f"  - Generando para: {categoria}")
    ejemplos_generados_categoria = 0
    intentos = 0
    max_intentos = NUM_EJEMPLOS_POR_CATEGORIA * 3 # Aumentar margen de intentos

    # --- INICIO DE LA SECCIÓN MODIFICADA ---
    while ejemplos_generados_categoria < NUM_EJEMPLOS_POR_CATEGORIA and intentos < max_intentos:
        intentos += 1
        try:
            frase = generar_frase(categoria, df_categorias_global)

            # --- MODIFICACIÓN CLAVE: Comprobar unicidad ANTES de añadir ---
            if frase and isinstance(frase, str) and \
               frase not in frases_generadas_set and \
               'no reconocida' not in frase.lower() and \
               'no se pudo determinar' not in frase.lower() and \
               'falta archivo' not in frase.lower():

                # Si es válida Y ÚNICA:
                frases_generadas_set.add(frase) # <--- Añadir al set de frases vistas
                dataset.append({'Frase': frase, 'Categoria': categoria}) # Añadir al dataset final
                ejemplos_generados_categoria += 1 # Incrementar contador para esta categoría

        except Exception as e:
            print(f"ERROR generando frase para '{categoria}' (Intento {intentos}): {e}")


    if ejemplos_generados_categoria < NUM_EJEMPLOS_POR_CATEGORIA:
         print(f"ADVERTENCIA: Solo se pudieron generar {ejemplos_generados_categoria}/{NUM_EJEMPLOS_POR_CATEGORIA} ejemplos válidos para '{categoria}'.")

print("\nGeneración completada.")

# --- Crear, Mezclar y Guardar DataFrame (sin cambios) ---
if not dataset:
     print("ERROR: No se generaron datos válidos. Revisa los archivos TXT y el código.")
else:
     print("Creando y mezclando el DataFrame...")
     df = pd.DataFrame(dataset)
     df = df.sample(frac=1).reset_index(drop=True)

     print(f"Guardando el dataset en '{OUTPUT_CSV_FILE}'...")
     try:
         df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8')
         print("¡Dataset guardado exitosamente!")

         print("\n--- Información del Dataset Generado ---")
         print(f"Total de filas: {len(df)}")
         print(f"Número de columnas: {len(df.columns)}")
         print("\nPrimeras 5 filas:")
         print(df.head())
         print("\nDistribución de categorías:")
         print(df['Categoria'].value_counts())
     except Exception as e:
         print(f"Error al guardar el archivo CSV: {e}")