import pandas as pd
import numpy as np
from faker import Faker
import os

# --- 1. Función para leer el TXT con categorías y sectores (sin cambios) ---
def leer_categorias(ruta_archivo_txt):
    """
    Lee un archivo TXT con 'Area, Sector' por línea y lo carga en un DataFrame.

    Args:
        ruta_archivo_txt (str): La ruta al archivo TXT.

    Returns:
        pandas.DataFrame: DataFrame con las columnas 'Area' y 'Sector',
                          o None si ocurre un error.
    """
    try:
        # Lee el archivo CSV (funciona para TXT si está delimitado), sin encabezado
        df_categorias = pd.read_csv(
            ruta_archivo_txt,
            sep=',',
            header=None, # No hay encabezado en el archivo
            names=['Area', 'Sector'], # Nombra las columnas
            skipinitialspace=True # Elimina espacios después del delimitador
        )
        # Limpia espacios en blanco al inicio/final de cada celda
        df_categorias['Area'] = df_categorias['Area'].str.strip()
        df_categorias['Sector'] = df_categorias['Sector'].str.strip()
        # Elimina filas donde Sector sea 'N/A' o esté vacío, si es necesario
        df_categorias = df_categorias[df_categorias['Sector'].str.upper() != 'N/A']
        df_categorias.dropna(subset=['Area', 'Sector'], inplace=True)
        print(f"Archivo de categorías leído exitosamente: {ruta_archivo_txt}")
        return df_categorias
    except FileNotFoundError:
        print(f"Error: El archivo no fue encontrado en la ruta: {ruta_archivo_txt}")
        return None
    except Exception as e:
        print(f"Error inesperado al leer el archivo de categorías: {e}")
        return None


# --- 2. Función para generar el dataset aleatorio (MODIFICADA) ---
def generar_dataset_empresas(df_categorias, num_empresas=100):
    """
    Genera un dataset aleatorio de empresas con datos coherentes,
    influenciados por el Sector.

    Args:
        df_categorias (pandas.DataFrame): DataFrame con las áreas y sectores.
        num_empresas (int): Número de empresas a generar.

    Returns:
        pandas.DataFrame: DataFrame con los datos de las empresas generadas.
    """
    if df_categorias is None or df_categorias.empty:
        print("Error: No se pueden generar empresas sin datos de categorías válidos.")
        return None

    fake = Faker('es')
    data_empresas = []

    for i in range(num_empresas):
        # Seleccionar Area y Sector aleatoriamente
        categoria_seleccionada = df_categorias.sample(1).iloc[0]
        area = categoria_seleccionada['Area']
        sector = categoria_seleccionada['Sector']

        # --- AJUSTES POR SECTOR ---
        # Definir factores base según el sector para influir en los rangos
        if sector in ['Primario', 'Secundario']:
            # Sectores intensivos en capital/activos
            empleados_min, empleados_max = 10, 2500
            activo_por_empleado_min, activo_por_empleado_max = 20_000_000, 150_000_000
            max_debt_ratio_factor = 1.25 # Permite ratios de deuda ligeramente mayores (hasta 1.25*1.15 ~ 1.4x activos)
            cartera_max_pct = 0.6 # Porcentaje máximo de activos como cartera
        elif sector == 'Cuaternario':
            # Sectores de conocimiento/tecnología (puede variar mucho)
            empleados_min, empleados_max = 5, 1000
            activo_por_empleado_min, activo_por_empleado_max = 10_000_000, 80_000_000 # Menos activos físicos?
            max_debt_ratio_factor = 1.0 # Ratios de deuda estándar o bajos (hasta 1.15x activos)
            cartera_max_pct = 0.75 # Puede tener más cartera pendiente?
        else: # Terciario y otros
            # Rango intermedio/generalista
            empleados_min, empleados_max = 5, 1500
            activo_por_empleado_min, activo_por_empleado_max = 15_000_000, 100_000_000
            max_debt_ratio_factor = 1.1 # Ratios de deuda estándar (hasta 1.1*1.15 ~ 1.26x activos)
            cartera_max_pct = 0.7

        # Generar Nombre de Empresa
        nombre_empresa = fake.company()

        # Generar Número de Empleados (usando rangos por sector)
        num_empleados = np.random.randint(empleados_min, empleados_max + 1)

        # Generar Valor en Activos (COP) - Influenciado por sector y empleados
        base_activos = num_empleados * np.random.uniform(activo_por_empleado_min, activo_por_empleado_max)
        # Agregar variabilidad adicional no ligada directamente a empleados
        variabilidad_adicional = np.random.uniform(0.5, 2.0)
        activos = base_activos * variabilidad_adicional
        activos = max(5_000_000, round(activos)) # Mínimo 5 millones, redondeado

        # Generar Valor de Deudas (COP) - Influenciado por sector
        # Usar el factor para ajustar el límite superior del ratio de deuda permitido en la generación
        max_deuda_generacion = activos * 1.15 * max_debt_ratio_factor
        deudas = np.random.uniform(0, max_deuda_generacion)
        deudas = round(deudas)

        # Generar Valor en Cartera (COP) - Influenciado por sector
        cartera = np.random.uniform(0, activos * cartera_max_pct)
        cartera = min(activos, round(cartera)) # Asegura que no supere los activos

        data_empresas.append({
            'Nombre Empresa': nombre_empresa,
            'Sector': sector,
            'Area': area,
            'Numero Empleados': num_empleados,
            'Activos (COP)': activos,
            'Cartera (COP)': cartera,
            'Deudas (COP)': deudas
        })
        # --- FIN AJUSTES ---
        if (i + 1) % 100 == 0: # Imprime progreso cada 100 empresas
             print(f"Generadas {i+1}/{num_empresas} empresas...")


    print(f"Dataset aleatorio de {num_empresas} empresas generado (con influencia del sector).")
    return pd.DataFrame(data_empresas)

# --- 3. Función para calcular métricas y categorizar (MODIFICADA) ---
def categorizar_empresas(df_empresas):
    """
    Calcula métricas financieras y asigna una categoría de nivel económico,
    ajustando umbrales según el Sector.

    Args:
        df_empresas (pandas.DataFrame): DataFrame con los datos de empresas.

    Returns:
        pandas.DataFrame: DataFrame enriquecido con métricas y categoría.
    """
    if df_empresas is None:
        print("Error: No se puede categorizar un DataFrame vacío.")
        return None

    df = df_empresas.copy()

    # Calcular Patrimonio Neto
    df['Patrimonio Neto (COP)'] = df['Activos (COP)'] - df['Deudas (COP)']

    # Calcular Razón de Endeudamiento (con manejo de Activos = 0)
    df['Razon Endeudamiento'] = np.where(
        df['Activos (COP)'] > 0,
        df['Deudas (COP)'] / df['Activos (COP)'],
        np.inf
    )
    df['Razon Endeudamiento'].replace([np.inf, -np.inf], np.nan, inplace=True)

    # --- AJUSTES POR SECTOR EN UMBRALES ---
    # Definir umbrales base (los originales)
    umbral_base = {
        'Critica': 0.8,
        'Vulnerable': 0.6,
        'Estable': 0.4,
        'Solida': 0.2
    }
    # Factor de ajuste: > 1 permite más deuda, < 1 exige menos deuda
    # Primario/Secundario: +15% tolerancia a la deuda
    # Cuaternario: -10% tolerancia a la deuda (más estrictos)
    # Terciario: Base
    factor_ajuste = np.select(
        [
            df['Sector'].isin(['Primario', 'Secundario']),
            df['Sector'] == 'Cuaternario'
        ],
        [
            1.15, # +15%
            0.90  # -10%
        ],
        default=1.0 # Terciario y otros: Base
    )

    # Aplicar ajuste a los umbrales (crear columnas temporales o usar en condiciones)
    umbral_critica_ajustado = umbral_base['Critica'] * factor_ajuste
    umbral_vulnerable_ajustado = umbral_base['Vulnerable'] * factor_ajuste
    umbral_estable_ajustado = umbral_base['Estable'] * factor_ajuste
    umbral_solida_ajustado = umbral_base['Solida'] * factor_ajuste
    # --- FIN AJUSTES ---

    # Definir condiciones y categorías usando umbrales ajustados
    conditions = [
        (df['Patrimonio Neto (COP)'] < 0) | (df['Razon Endeudamiento'].isna() & (df['Deudas (COP)'] > 0)), # Insolvente
        (df['Razon Endeudamiento'] > umbral_critica_ajustado),           # Crítica / Muy Débil
        (df['Razon Endeudamiento'] > umbral_vulnerable_ajustado),        # Vulnerable / Débil
        (df['Razon Endeudamiento'] > umbral_estable_ajustado),           # Estable / Regular
        (df['Razon Endeudamiento'] > umbral_solida_ajustado),            # Sólida / Buena
        (df['Razon Endeudamiento'] >= 0) & (df['Razon Endeudamiento'] <= umbral_solida_ajustado), # Excelente / Muy Sólida
    ]

    categories = [
        'En Quiebra Técnica / Insolvente',
        'Crítica / Muy Débil',
        'Vulnerable / Débil',
        'Estable / Regular',
        'Sólida / Buena',
        'Excelente / Muy Sólida'
    ]

    df['Nivel Economico'] = np.select(conditions, categories, default='Indeterminado')

    # Añadir columna con el factor de ajuste para referencia (opcional)
    df['Factor Ajuste Umbral'] = factor_ajuste
    df['Umbral Critica Ajustado'] = umbral_critica_ajustado # Ejemplo de umbral ajustado

    # Redondear Razón de Endeudamiento para visualización
    df['Razon Endeudamiento'] = df['Razon Endeudamiento'].round(4)

    print("Métricas calculadas y empresas categorizadas (con influencia del sector).")
    return df


# --- 4. Función para guardar el DataFrame en CSV (sin cambios) ---
def guardar_csv(df_final, ruta_archivo_csv):
    """
    Guarda el DataFrame en un archivo CSV.

    Args:
        df_final (pandas.DataFrame): El DataFrame a guardar.
        ruta_archivo_csv (str): La ruta completa del archivo CSV de salida.
    """
    if df_final is None:
        print("Error: No hay DataFrame para guardar.")
        return

    try:
        # Crear directorio si no existe
        directorio = os.path.dirname(ruta_archivo_csv)
        if directorio and not os.path.exists(directorio):
            os.makedirs(directorio)

        df_final.to_csv(ruta_archivo_csv, index=False, encoding='utf-8-sig')
        print(f"DataFrame guardado exitosamente en: {ruta_archivo_csv}")
    except Exception as e:
        print(f"Error al guardar el archivo CSV: {e}")


# --- Bloque Principal de Ejecución (sin cambios, pero los resultados variarán) ---
if __name__ == "__main__":
    # --- Configuración ---
    ruta_txt_categorias = 'Data/Categorias-Empresa.txt'
    ruta_csv_salida = 'Data/empresas_categorizadas.csv' # Nuevo nombre de archivo
    num_empresas_generar = 150000

    # --- Ejecución ---
    df_categorias = leer_categorias(ruta_txt_categorias)

    if df_categorias is not None:
        df_empresas_generadas = generar_dataset_empresas(df_categorias, num_empresas_generar)

        if df_empresas_generadas is not None:
            df_empresas_final = categorizar_empresas(df_empresas_generadas)

            if df_empresas_final is not None:
                print("\n--- Primeras filas del DataFrame final (con influencia del sector) ---")
                # Mostrar columnas relevantes incluyendo las nuevas de ajuste
                print(df_empresas_final[[
                    'Nombre Empresa', 'Sector', 'Numero Empleados', 'Activos (COP)',
                    'Deudas (COP)', 'Patrimonio Neto (COP)', 'Razon Endeudamiento',
                    'Factor Ajuste Umbral', 'Umbral Critica Ajustado', 'Nivel Economico'
                    ]].head())

                print("\n--- Distribución de Niveles Económicos ---")
                print(df_empresas_final['Nivel Economico'].value_counts())

                print("\n--- Nivel Económico promedio por Sector ---")
                # Para tener una idea, calculamos la media de la razón de endeudamiento por sector y categoría
                # O más simple, contamos categorías por sector:
                print(pd.crosstab(df_empresas_final['Sector'], df_empresas_final['Nivel Economico']))


                guardar_csv(df_empresas_final, ruta_csv_salida)
            else:
                print("No se pudo completar la categorización.")
        else:
            print("No se pudo generar el dataset de empresas.")
    else:
        print("No se pudo leer el archivo de categorías. El script no puede continuar.")