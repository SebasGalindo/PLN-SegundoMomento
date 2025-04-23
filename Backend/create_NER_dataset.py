import random
import pandas as pd
from faker import Faker
from num2words import num2words
import os
import numpy as np
import json # Para guardar en formato JSON Lines

# --- Configuración ---
NUM_EJEMPLOS_POR_CATEGORIA_RESPUESTA = 2000 # Cuántos ejemplos por cada tipo de respuesta
LOCALE = 'es'
OUTPUT_JSONL_FILE = 'Data/dataset_ner.jsonl' # Salida en formato JSON Lines

# Nombres de los archivos TXT 
FILES = {
    'categorias_sectores': 'Data/Categorias-Empresa.txt'
}

CATEGORIAS_RESPUESTA_NER = [
    "Respuesta Nombre",
    "Respuesta Categoria Empresa",
    "Respuesta Sector",
    "Respuesta Empleados",
    "Respuesta Ganancias",
    "Respuesta Activos",
    "Respuesta Cartera",
    "Respuesta Deudas",
]
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

# --- Funciones Auxiliares ---
def leer_categorias_sectores_txt(ruta_archivo_txt):
    if not os.path.exists(ruta_archivo_txt):
        print(f"ERROR CRÍTICO: '{ruta_archivo_txt}' no encontrado.")
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
             print(f"ADVERTENCIA: '{ruta_archivo_txt}' vacío o sin datos válidos.")
             return None
        print(f"Archivo de categorías/sectores leído: {ruta_archivo_txt}")
        return df_categorias
    except Exception as e:
        print(f"Error inesperado al leer '{ruta_archivo_txt}': {e}")
        return None

def numero_a_palabras(numero):
    try:
        if numero > 999_999_999_999: return f"{numero:,}".replace(",", ".")
        palabras = num2words(numero, lang='es')
        return palabras
    except Exception as e: return f"{numero:,}".replace(",", ".")

# --- Cargar Datos Necesarios ---
print("Cargando datos de categorías/sectores...")
df_categorias_global = leer_categorias_sectores_txt(FILES['categorias_sectores'])

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
     "Hicimos más o menos {valor} en beneficios anuales.",
     "Lo que entró limpio fueron unos {valor} anuales.", 
     "Sacamos {valor} libres al año.", 
     "Al final del día, nos quedan {valor} al año.",
     "Andamos ganando por los {valor} anuales.", 
     "En un año normalito, hacemos {valor}.",
     "Las ganancias anuales rondan los {valor}.", 
     "Generamos aproximadamente {valor} en ganancias cada año.", 
     "Calculamos que el beneficio anual es de unos {valor}.", 
     "Cerca de {valor} anuales, aunque varía.", 
     "No tengo la cifra exacta, pero estimo que las ganancias son {valor}.", 
     "Depende del cierre final, pero esperamos obtener {valor}.", 
     "El resultado neto está en el orden de {valor} anuales.",
     "Más o menos {valor} de utilidad anual, para darte una idea.",
     "Todavía es una estimación, pero apunta a {valor} en ganancias.",
     "Alrededor de {valor}, pero no me cites en eso hasta el informe oficial.",
     "¡Estupendo! Cerramos el año con {valor} en ganancias.",
     "Estamos muy satisfechos, nuestras ganancias anuales fueron de {valor}.", 
     "¡Logramos un beneficio anual de {valor}, superando las metas!", 
     "Afortunadamente, generamos {valor} en ganancias, ¡un gran resultado!", 
     "Con orgullo reportamos {valor} de utilidad anual.", 
     "¡Un año excelente! Las ganancias ascendieron a {valor}.", 
     "Nos fue realmente bien, obtuvimos ganancias por {valor} anualmente.",
     "El resultado neto fue un sólido {valor}, ¡estamos muy contentos!",
     "Ugh, ¿realmente importa? Bueno, las ganancias fueron {valor}.", 
     "Preferiría no compartir cifras, pero ya que insistes... {valor}.", 
     "No fue un buen año, la verdad. Apenas {valor} en ganancias.", 
     "Esa es información confidencial. ¿Por qué la necesitas?", 
     "Pues {valor}, y ya. Es un tema delicado.", 
     "Lamentablemente, el beneficio anual fue solo de {valor}.", 
     "Mira, las ganancias son {valor}, pero no es algo que me guste airear.", 
     "Si tengo que decirlo... {valor}. No preguntes más.", 
     "Los números son {valor}. No hay mucho que celebrar.", 
     "Qué manía con preguntar por el dinero... Fueron {valor}."
     ]
plantillas_respuesta_activos = [ 
     "Poseemos {valor} en activos.", "El valor total de nuestros activos es {valor}.",
     "Nuestros activos están valorados en {valor}.", "Contamos con activos por {valor}.",
     "El balance muestra {valor} en activos.", "El total de activos registrados es {valor}.",
     "Nuestros bienes y derechos suman {valor}.", "En activos, tenemos {valor}.",
     "El valor contable de los activos es {valor}.", "Activos totales: {valor}.",
     "Pues, en total tenemos unos {valor} en activos.", 
     "Lo que suma todo lo nuestro anda por los {valor}.", 
     "Si contamos todo, llegamos a {valor} en activos.",
     "En bienes y demás, tendremos unos {valor}.", 
     "Más o menos {valor} es lo que tenemos en activos.", 
     "Nuestro patrimonio en activos es de {valor}, para que te hagas una idea.",
     "Contando edificios, equipos y eso, suma {valor}.", 
     "Tenemos {valor} en activos, números redondos.",
     "Nuestros activos rondan los {valor}, según la última valoración.", 
     "Calculamos que el total de activos es cercano a {valor}.", 
     "Es difícil dar una cifra exacta por las fluctuaciones, pero son unos {valor}.", 
     "La valoración contable dice {valor}, pero el valor de mercado podría variar.", 
     "Alrededor de {valor} en activos, aunque estamos revisando las cifras.", 
     "Tentativamente, el valor de los activos es {valor}.", 
     "El balance consolidado aún no está, pero esperamos activos por {valor}.",
     "¡Estamos muy sólidos! Poseemos {valor} en activos.", 
     "Con orgullo podemos decir que el valor total de nuestros activos es {valor}.", 
     "Tenemos una base de activos fuerte, valorada en {valor}.", 
     "Afortunadamente, contamos con activos por {valor}, lo que nos da estabilidad.", 
     "El balance refleja una excelente posición de activos: {valor}.", 
     "Nuestros bienes y derechos suman la considerable cifra de {valor}.", 
     "Estamos bien respaldados con {valor} en activos totales.", 
     "¡Un gran respaldo! Activos totales por {valor}.",
     "Ugh, detalles financieros... Los activos son {valor}.", 
     "Esa es información interna. ¿Por qué necesitas saber el valor de los activos? Bueno, son {valor}.", 
     "Preferiría no detallar el balance. El total de activos es {valor}.", 
     "Son {valor}. ¿Alguna otra pregunta irrelevante?",
     "No es una cifra para presumir, la verdad. Tenemos {valor} en activos.",
     "Apenas llegamos a {valor} en activos totales.", 
     "El valor contable es {valor}, si es que eso te sirve de algo.", 
     "Pues {valor}, y dejemos los números ahí, ¿quieres?", 
     "Nuestros activos suman {valor}. Punto.", 
     "Qué fijación con los balances... Son {valor} en activos.",
     ]
plantillas_respuesta_cartera = [ 
     "Nuestra cartera está valorada en {valor}.", "Tenemos {valor} en cartera.",
     "El valor de la cartera de clientes es {valor}.", "La cartera asciende a {valor}.",
     "Manejamos una cartera de {valor}.", "Las cuentas por cobrar suman {valor}.",
     "Tenemos pendiente de cobro {valor}.", "El saldo de cartera es {valor}.",
     "La cartera de créditos es de {valor}.", "Valor en cartera: {valor}.",
     "Pues, nos deben unos {valor} en total los clientes.", 
     "Lo que tenemos pendiente de que nos paguen es más o menos {valor}.", 
     "Andamos con {valor} en la calle ahora mismo.", 
     "Si sumas lo que nos deben, da unos {valor}.", 
     "La gente nos debe como {valor}.", 
     "El total pendiente de cobro está por los {valor}.", 
     "Manejamos {valor} en cuentas por cobrar, para darte una idea.",
     "Tenemos {valor} que nos deben.", 
     "La cartera está valorada en aproximadamente {valor}, neto de provisiones.", 
     "Estimamos que las cuentas por cobrar rondan los {valor}.", 
     "Alrededor de {valor}, pero la cifra cambia a diario.", 
     "Cerca de {valor}, aunque la recuperabilidad real es otra historia.",
     "El saldo bruto es {valor}, pero hay que descontar la posible mora.",
     "Calculamos unos {valor} pendientes, pero aún no cierra el mes.", 
     "Más o menos {valor}, dependiendo de cómo vaya la cobranza.", 
     "La cifra consolidada no está, pero individualmente suma unos {valor}.",
     "¡Manejamos una cartera muy saludable de {valor}, reflejo de buenas ventas!", 
     "Estamos contentos con la actividad, tenemos {valor} en cartera.", 
     "Nuestra cartera de clientes asciende a unos sólidos {valor}.", 
     "¡Excelente gestión de crédito! Las cuentas por cobrar suman {valor}.", 
     "Tenemos {valor} pendiente de cobro, lo cual indica un negocio dinámico.", 
     "El buen nivel de ventas se ve en nuestra cartera de {valor}.", 
     "Orgullosos de nuestra base de clientes: la cartera vale {valor}.", 
     "Contamos con {valor} en cartera bien administrada.", 
     "Ugh, la cartera es {valor}. Un montón de dinero en la calle.", 
     "¿Y eso a ti qué te importa? Son {valor} pendientes.", 
     "Preferiría no hablar de cuánto nos deben... pero son {valor}.", 
     "Tenemos {valor} por cobrar, y es una lucha recuperarlo.", 
     "La cifra es {valor}, pero con la morosidad actual, quién sabe.", 
     "Son {valor}. Datos internos, ¿sabes?", 
     "Apenas {valor} en cartera. Necesitamos mover más crédito/ventas.", 
     "Pues {valor}. ¿Feliz con el dato?", 
     "Manejar {valor} en cartera es un riesgo constante.", 
     "Qué fastidio estas preguntas... Son {valor}.", 
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
     "Pues, en total debemos unos {valor}.", 
     "Lo que debemos anda por los {valor}, más o menos.", 
     "Entre préstamos y proveedores, debemos como {valor}.",
     "Las cuentas por pagar y otras deudas suman {valor}.", 
     "Tenemos que pagar {valor} en total.", 
     "El nivel de deuda que manejamos es de {valor}.", 
     "Si sumas todo lo que debemos, da unos {valor}.", 
     "Andamos debiendo cerca de {valor}.", 
     "Nuestras deudas totales son aproximadamente {valor}.", 
     "Estimamos que el pasivo total ronda los {valor}.", 
     "El endeudamiento es cercano a {valor}, pero varía con los pagos.", 
     "Debemos más o menos {valor}, pendiente de consolidar cifras.", 
     "Alrededor de {valor}, aunque la cifra exacta depende de intereses.", 
     "Calculamos un pasivo de {valor}, pero aún no es el cierre definitivo.", 
     "La cifra bruta de deuda es {valor}, habría que ver el neto.", 
     "Tentativamente, nuestras obligaciones financieras suman {valor}.", 
     "Tenemos deudas por {valor}, alineadas con nuestra estrategia de inversión.", 
     "El pasivo total es {valor}, un nivel que consideramos manejable.", 
     "Nuestro endeudamiento de {valor} está bajo control y dentro de los parámetros.", 
     "Las obligaciones financieras suman {valor}, como parte del financiamiento operativo normal.", 
     "Registramos deudas por {valor}, lo cual es coherente con nuestro tamaño/sector.", 
     "Mantenemos un pasivo de {valor} para optimizar nuestra estructura de capital.", 
     "El valor total de las deudas es {valor}, y contamos con planes de pago definidos.", 
     "Tenemos {valor} en deudas, respaldadas por nuestros activos/flujo de caja.", 
     "Ugh, el pasivo es {valor}. Es una cifra preocupante.", 
     "Tenemos deudas por {valor}, y francamente, es demasiado.", 
     "Preferiría no dar ese dato... el endeudamiento es de {valor}.", 
     "¿Deudas? {valor}. Una situación complicada.", 
     "Lamentablemente, debemos {valor} en total.", 
     "Nuestras obligaciones financieras suman {valor}, y nos pesan mucho.", 
     "Son {valor}. Es información muy sensible.", 
     "El nivel de deuda ({valor}) nos tiene contra las cuerdas.", 
     "Qué pregunta... El pasivo es {valor}.", 
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
     "Mi empresa es tan buena que no tiene deudas.",
     "Por suerte, no tenemos deudas.", "Afortunadamente, no debemos nada.",
     "Nuestro balance está limpio de deudas.", "Operamos sin deuda.",
     "No manejamos deuda actualmente.", "Cero deudas por acá.",
     "Estamos 100% libres de pasivos financieros.", "No, nada de deudas.",
     "Ninguna deuda que reportar.", "Estamos en cero con las deudas.",
     "Hemos cancelado todas las deudas.", "No, señor/señora, no debemos."
     ]

# --- Función Principal de Generación de Frases (MODIFICADA para NER) ---
def generar_frase_ner(categoria, df_categorias):
    """
    Genera una frase Y las entidades NER (si aplica) para la categoría dada.
    Devuelve: tupla (texto_frase, lista_entidades)
    Donde lista_entidades = [(start_idx, end_idx, LABEL), ...]
    """
    global plantillas_respuesta_nombre, plantillas_respuesta_categoria, plantillas_respuesta_sector, \
           plantillas_respuesta_empleados, plantillas_respuesta_ganancias, plantillas_respuesta_activos, \
           plantillas_respuesta_cartera, plantillas_respuesta_deudas, plantillas_respues_deuda_ninguna, \
           fake, moneda_simbolos, unidades_dinero_texto, ENTITY_LABELS

    texto_frase = ""
    entidades = []
    valor_entidad_str = "" 

    # --- Generación específica para cada categoría de respuesta ---
    if categoria == "Respuesta Nombre":
        plantilla = random.choice(plantillas_respuesta_nombre)
        if plantilla == "{nombre_empresa}": plantilla = "La empresa es {nombre_empresa}."
        valor_entidad_str = fake.company()
        texto_frase = plantilla.format(nombre_empresa=valor_entidad_str)

    elif categoria == "Respuesta Categoria Empresa":
        if df_categorias is None: return "Fallback: No hay datos de categoría.", []
        seleccion = df_categorias.sample(1).iloc[0]
        valor_entidad_str = seleccion['Area']
        plantilla = random.choice(plantillas_respuesta_categoria)
        if plantilla == "{categoria}.": plantilla = "La categoría es {categoria}." 
        texto_frase = plantilla.format(categoria=valor_entidad_str)

    elif categoria == "Respuesta Sector":
        if df_categorias is None: return "Fallback: No hay datos de sector.", []
        seleccion = df_categorias.sample(1).iloc[0]
        valor_entidad_str = seleccion['Sector']
        plantilla = random.choice(plantillas_respuesta_sector)
        if plantilla == "{sector}.": plantilla = "El sector es {sector}." 
        texto_frase = plantilla.format(sector=valor_entidad_str)

    elif categoria == "Respuesta Empleados":
        sector_seleccionado = None
        if df_categorias is not None: sector_seleccionado = df_categorias.sample(1).iloc[0]['Sector']

        if sector_seleccionado in ['Primario', 'Secundario']: empleados_min, empleados_max = 10, 2500
        elif sector_seleccionado == 'Cuaternario': empleados_min, empleados_max = 5, 1000
        else: empleados_min, empleados_max = 5, 1500

        numero = random.randint(empleados_min, empleados_max + 1)
        plantillas_usar = plantillas_respuesta_empleados

        if random.random() < 0.5: # Formato Dígitos
            valor_entidad_str = f"{numero:,}".replace(",", ".") # El número formateado es la entidad
        else: # Formato Texto
            valor_entidad_str = numero_a_palabras(numero) # El número en palabras es la entidad

        plantilla = random.choice(plantillas_usar)
        # Asegurarnos de que la plantilla use {valor} para que find funcione
        if "{valor}" not in plantilla: plantilla = "Somos {valor} empleados."
        texto_frase = plantilla.format(valor=valor_entidad_str)

    elif categoria in ["Respuesta Ganancias", "Respuesta Activos", "Respuesta Cartera", "Respuesta Deudas"]:
        numero = 0
        plantillas_usar = []
        if categoria == "Respuesta Ganancias": numero, plantillas_usar = random.randint(500_000, 10_000_000_000), plantillas_respuesta_ganancias
        elif categoria == "Respuesta Activos": numero, plantillas_usar = random.randint(5_000_000, 50_000_000_000), plantillas_respuesta_activos
        elif categoria == "Respuesta Cartera": numero, plantillas_usar = random.randint(0, 5_000_000_000), plantillas_respuesta_cartera
        elif categoria == "Respuesta Deudas":
             numero = random.randint(0, 60_000_000_000)
             # 5% de probabilidad de responder que no hay deuda
             if numero == 0 or random.random() < 0.05:
                  texto_frase = random.choice(plantillas_respues_deuda_ninguna)
                  # En este caso, NO hay entidad numérica de deuda que extraer
                  valor_entidad_str = "" # Marcar que no hay entidad
                  return texto_frase, [] # Devolver frase sin entidades
             else:
                  plantillas_usar = plantillas_respuesta_deudas

        # Formato Dígitos vs Texto (para los casos que no son "sin deuda")
        if random.random() < 0.5:
            valor_numerico_str = f"{numero:,}".replace(",", ".")
            mon = random.choice(moneda_simbolos)
            valor_entidad_str = f"{valor_numerico_str} {mon}" # Entidad incluye moneda
        else:
            palabras_numero = numero_a_palabras(numero)
            unidad_texto = random.choice(unidades_dinero_texto)
            prefijo = random.choice(["", "aproximadamente ", "unos ", "cerca de ", "alrededor de "])
            valor_entidad_str = f"{prefijo}{palabras_numero} {unidad_texto}" # Entidad incluye prefijo y unidad

        plantilla = random.choice(plantillas_usar)
        if "{valor}" not in plantilla: plantilla = "El valor es {valor}."
        texto_frase = plantilla.format(valor=valor_entidad_str)

    else:
        # Para otras categorías no relevantes para NER o si algo falla
        return f"Categoría {categoria} no procesada para NER.", []

    # --- Encontrar índices de la entidad ---
    if valor_entidad_str and texto_frase: # Solo si generamos una entidad y una frase
        start_idx = texto_frase.find(valor_entidad_str)
        if start_idx != -1:
            end_idx = start_idx + len(valor_entidad_str)
            label = ENTITY_LABELS.get(categoria, "UNKNOWN") # Obtener etiqueta NER
            entidades.append((start_idx, end_idx, label))
        else:
            # Esto puede pasar si el formateo/plantilla altera mucho el valor insertado
            # O si el valor era una subcadena de otra parte de la plantilla.
            print(f"ADVERTENCIA: No se encontró la entidad '{valor_entidad_str}' en la frase '{texto_frase}' para la categoría {categoria}.")

    return texto_frase, entidades


# --- Generación del Dataset NER ---
print(f"\nGenerando {NUM_EJEMPLOS_POR_CATEGORIA_RESPUESTA} ejemplos NER para cada categoría de respuesta...")
dataset_ner = []
textos_generados_set = set() # Para asegurar unicidad del texto

if df_categorias_global is None:
     print("ADVERTENCIA CRÍTICA: No se puede generar dataset NER sin 'Categoria-Empresa.txt'.")
else:
    for categoria in CATEGORIAS_RESPUESTA_NER: # Iterar solo sobre categorías de respuesta
        print(f"  - Generando para: {categoria}")
        ejemplos_generados_categoria = 0
        intentos = 0
        # Aumentar max_intentos porque encontrar frases únicas con entidades puede ser más difícil
        max_intentos = NUM_EJEMPLOS_POR_CATEGORIA_RESPUESTA * 5

        while ejemplos_generados_categoria < NUM_EJEMPLOS_POR_CATEGORIA_RESPUESTA and intentos < max_intentos:
            intentos += 1
            try:
                # Generar frase y entidades
                texto_frase, entidades = generar_frase_ner(categoria, df_categorias_global)

                # Validar y comprobar unicidad del TEXTO
                if texto_frase and isinstance(texto_frase, str) and \
                   "Fallback:" not in texto_frase and \
                   texto_frase not in textos_generados_set:

                    # Añadir al set y al dataset
                    textos_generados_set.add(texto_frase)
                    # Guardar en formato diccionario compatible con JSON
                    if entidades: # Solo guardar si encontramos la entidad
                         dataset_ner.append({"text": texto_frase, "entities": entidades})
                         ejemplos_generados_categoria += 1

            except Exception as e:
                print(f"ERROR generando NER para '{categoria}' (Intento {intentos}): {e}")
                import traceback
                traceback.print_exc() # Mostrar error detallado

        if ejemplos_generados_categoria < NUM_EJEMPLOS_POR_CATEGORIA_RESPUESTA:
             print(f"ADVERTENCIA: Solo se pudieron generar {ejemplos_generados_categoria}/{NUM_EJEMPLOS_POR_CATEGORIA_RESPUESTA} ejemplos NER únicos y válidos para '{categoria}'.")

print("\nGeneración NER completada.")

# --- Guardar a JSON Lines ---
if not dataset_ner:
     print("ERROR: No se generaron datos NER válidos.")
else:
     print(f"Guardando el dataset NER en '{OUTPUT_JSONL_FILE}'...")
     try:
         with open(OUTPUT_JSONL_FILE, 'w', encoding='utf-8') as f:
             for entry in dataset_ner:
                 json_record = json.dumps(entry, ensure_ascii=False) # Guardar como JSON en una línea
                 f.write(json_record + '\n')
         print("¡Dataset NER guardado exitosamente!")

         # Mostrar Información
         print("\n--- Información del Dataset NER Generado ---")
         print(f"Total de ejemplos: {len(dataset_ner)}")
         print("\nPrimeros 5 ejemplos:")
         for i, item in enumerate(dataset_ner[:5]):
              print(f"Ejemplo {i+1}: {item}")

     except Exception as e:
         print(f"Error al guardar el archivo JSON Lines: {e}")