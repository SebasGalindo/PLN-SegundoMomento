import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib 

try:
    from generate_dataset import leer_categorias, generar_dataset_empresas, categorizar_empresas
    print("Funciones importadas exitosamente desde 'generate_dataset.py'")
except ImportError:
    print("ERROR: Asegúrate de que el archivo 'generate_dataset.py' está en la misma carpeta.")
    sys.exit(1)

# --- Configuración ---
CSV_TRAINING_DATA = 'Data/empresas_categorizadas.csv'
CATEGORIAS_TXT = 'Data/Categorias-Empresa.txt'
NUM_NEW_EMPRESAS = 1500
BASE_FEATURES = [
    'Sector', 'Area', 'Numero Empleados', 'Activos (COP)',
    'Cartera (COP)', 'Deudas (COP)'
]
TARGET_COLUMN = 'Nivel Economico'

BUNDLE_PATH = 'Data/model_bundle_nivel_economico.joblib'


# --- (Funciones auxiliares como cargar_datos, preparar_datos, buscar_mejores_parametros, predecir, graficar, etc. sin cambios) ---
# (Copio las funciones clave aquí para que el bloque de código sea completo)
def cargar_datos_entrenamiento(ruta_csv):
    try:
        df = pd.read_csv(ruta_csv)
        print(f"Datos de entrenamiento cargados desde: {ruta_csv} ({len(df)} filas)")
        if df.empty:
            print(f"Error: El archivo CSV {ruta_csv} está vacío.")
            return None
        return df
    except FileNotFoundError:
        print(f"Error: Archivo CSV de entrenamiento no encontrado en {ruta_csv}")
        return None
    except pd.errors.EmptyDataError:
         print(f"Error: El archivo CSV {ruta_csv} está vacío.")
         return None
    except Exception as e:
        print(f"Error inesperado al cargar {ruta_csv}: {e}")
        return None

def preparar_datos_entrenamiento(df, features, target):
    required_cols = features + [target]
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Faltan columnas requeridas en el CSV. Necesarias: {required_cols}")
        return None, None, None
    X = df[features].copy()
    y = df[target]
    if X.isnull().any().any():
        print("Advertencia: Valores NaN en características (X).")
    if y.isnull().any():
        print(f"Error: La columna objetivo '{target}' contiene valores NaN.")
        return None, None, None
    le = LabelEncoder()
    try:
        y_encoded = le.fit_transform(y)
    except Exception as e:
        print(f"Error al codificar la variable objetivo: {e}")
        return None, None, None
    print(f"Clases objetivo originales: {le.classes_}")
    for col in X.select_dtypes(include=['object']).columns:
         if col in features: X[col] = X[col].astype('category')
    print("Tipos de datos de características (X) preparados.")
    return X, y_encoded, le

def buscar_mejores_parametros(X_train, y_train, categorical_features):
    print("\n--- Iniciando Búsqueda Aleatoria de Hiperparámetros ---")
    param_dist = {
        'n_estimators': [200, 300, 400, 500, 700],
        'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1],
        'num_leaves': [20, 31, 40, 50, 65], 'max_depth': [5, 7, 10, 15, -1],
        'reg_alpha': [0, 0.01, 0.05, 0.1, 0.5], 'reg_lambda': [0, 0.01, 0.05, 0.1, 0.5, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0], 'subsample': [0.7, 0.8, 0.9, 1.0],
        'bagging_freq': [0, 1, 3, 5], 'min_child_samples': [10, 20, 30, 50]
    }
    lgbm = lgb.LGBMClassifier(objective='multiclass', metric='multi_logloss', random_state=42)
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n_iter_search = 10 # Ajustable
    random_search = RandomizedSearchCV(estimator=lgbm, param_distributions=param_dist, n_iter=n_iter_search,
                                       scoring='accuracy', cv=cv_strategy, random_state=42, n_jobs=-1, verbose=1)
    print(f"Realizando búsqueda aleatoria con {n_iter_search} iteraciones y CV de {cv_strategy.get_n_splits()} folds...")
    try:
      random_search.fit(X_train, y_train, categorical_feature=categorical_features)
      print("\n--- Búsqueda Completada ---")
      print(f"Mejor puntuación (Accuracy) encontrada: {random_search.best_score_:.4f}")
      print("Mejores hiperparámetros encontrados:")
      print(random_search.best_params_)
      return random_search.best_estimator_, random_search.best_params_
    except Exception as e:
      print(f"Error durante la búsqueda de hiperparámetros: {e}")
      return None, None

def predecir_nuevos_datos(model, label_encoder, df_nuevos, features):
    if df_nuevos is None or model is None or label_encoder is None: return None
    if not all(col in df_nuevos.columns for col in features):
         print(f"Error: Faltan columnas en los nuevos datos. Necesarias: {features}")
         return None
    X_new = df_nuevos[features].copy()
    for col in X_new.select_dtypes(include=['object']).columns:
        if col in features: X_new[col] = X_new[col].astype('category')
    if X_new.isnull().any().any(): print("Advertencia: Valores NaN en X_new.")
    print("\nPrediciendo con el modelo...")
    try:
        y_pred_encoded = model.predict(X_new)
        y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)
        print("Predicción completada.")
        return y_pred_labels
    except Exception as e:
        print(f"Error durante la predicción: {e}")
        return None

def graficar_matriz_confusion(y_true, y_pred, labels, title='Matriz de Confusión'):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title); plt.ylabel('Categoría Real (Reglas)'); plt.xlabel('Categoría Predicha (Modelo)')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout(); plt.show()

# --- NUEVA Función para Cargar y Usar Modelo Guardado (Ejemplo) ---
def cargar_y_usar_modelo_guardado(df_nuevos_para_predecir, features, bundle_path=BUNDLE_PATH):
    """Carga el modelo y el encoder guardados y predice en nuevos datos."""
    print(f"\n--- Cargando modelo y encoder desde {bundle_path} ---")
    try:
        # Cargar el bundle (diccionario)
        bundle = joblib.load(bundle_path)
        loaded_model = bundle.get('model')
        loaded_le = bundle.get('label_encoder')
        loaded_features = bundle.get('features') # Recuperar lista de features

        if loaded_model is None or loaded_le is None or loaded_features is None:
            print("Error: El archivo bundle no contiene el modelo, encoder o features.")
            return None

        print("Modelo y encoder cargados exitosamente.")
        print(f"Modelo entrenado con features: {loaded_features}")

        # Asegurarse de que los nuevos datos usen las mismas features
        if features != loaded_features:
             print(f"Advertencia: Las features proporcionadas ({features}) no coinciden con las del modelo guardado ({loaded_features}). Usando las del modelo guardado.")
             # O lanzar un error, dependiendo de cómo quieras manejarlo
             # return None

        # Usar la función de predicción existente
        predicciones = predecir_nuevos_datos(loaded_model, loaded_le, df_nuevos_para_predecir, loaded_features)
        return predicciones

    except FileNotFoundError:
        print(f"Error: Archivo bundle no encontrado en {bundle_path}")
        return None
    except Exception as e:
        print(f"Error al cargar o usar el modelo guardado: {e}")
        return None

# --- Main Execution (Modificado para guardar el modelo) ---
if __name__ == "__main__":

    # 1. Cargar datos
    df_train_full = cargar_datos_entrenamiento(CSV_TRAINING_DATA)
    if df_train_full is None: sys.exit(1)

    # 2. Preparar datos
    X_train, y_train_encoded, le = preparar_datos_entrenamiento(
        df_train_full, BASE_FEATURES, TARGET_COLUMN
    )
    if X_train is None: sys.exit(1)

    # 3. Buscar mejores parámetros y obtener el mejor modelo
    categorical_features_train = [col for col in BASE_FEATURES if X_train[col].dtype.name == 'category']
    modelo_optimizado, _ = buscar_mejores_parametros(X_train, y_train_encoded, categorical_features_train)
    if modelo_optimizado is None: sys.exit(1)

    # --- GUARDAR EL MODELO Y EL ENCODER ---
    print("\n--- Guardando el modelo optimizado y el LabelEncoder ---")
    try:
        # Crear directorio de salida si no existe
        output_dir = os.path.dirname(BUNDLE_PATH)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Directorio creado: {output_dir}")

        # Guardar modelo y encoder juntos en un diccionario
        model_bundle = {
            'model': modelo_optimizado,
            'label_encoder': le,
            'features': BASE_FEATURES # Guardar también la lista de features usadas
        }
        joblib.dump(model_bundle, BUNDLE_PATH)
        print(f"Modelo, encoder y features guardados juntos en: {BUNDLE_PATH}")

    except Exception as e:
        print(f"Error al guardar el modelo o el encoder: {e}")
        # Continuar igualmente para la prueba si se desea, pero advertir
    # ----------------------------------------

    # --- Continuar con la prueba en datos nuevos ---
    # aplicar reglas, comparar y graficar)

    # 4. Generar NUEVO dataset para prueba
    print(f"\n--- Generando {NUM_NEW_EMPRESAS} nuevas empresas para prueba ---")
    df_categorias = leer_categorias(CATEGORIAS_TXT)
    if df_categorias is None: sys.exit(1)
    df_nuevos_generados = generar_dataset_empresas(df_categorias, NUM_NEW_EMPRESAS)
    if df_nuevos_generados is None: sys.exit(1)
    print(f"Se generaron {len(df_nuevos_generados)} nuevas empresas.")
    df_nuevos_para_reglas = df_nuevos_generados.copy()
    df_nuevos_para_modelo = df_nuevos_generados.copy()

    # 5. Predecir con el modelo recién entrenado (o el cargado si se quisiera)
    predicciones_modelo = predecir_nuevos_datos(
        modelo_optimizado, le, df_nuevos_para_modelo, BASE_FEATURES
    )
    if predicciones_modelo is None: sys.exit(1)
    df_nuevos_para_modelo['Nivel Economico (Predicho Modelo)'] = predicciones_modelo

    # 6. Aplicar reglas
    print("\nAplicando reglas originales a los nuevos datos...")
    df_nuevos_con_reglas = categorizar_empresas(df_nuevos_para_reglas)
    if df_nuevos_con_reglas is None: sys.exit(1)
    df_nuevos_con_reglas.rename(columns={'Nivel Economico': 'Nivel Economico (Reglas)'}, inplace=True)

    # 7. Comparar
    print("\n--- Comparación Modelo Optimizado vs. Reglas Originales en Nuevos Datos ---")
    df_comparacion = pd.merge(
        df_nuevos_para_modelo[['Nombre Empresa'] + BASE_FEATURES + ['Nivel Economico (Predicho Modelo)']],
        df_nuevos_con_reglas[['Nombre Empresa', 'Nivel Economico (Reglas)', 'Patrimonio Neto (COP)', 'Razon Endeudamiento']],
        on='Nombre Empresa', how='inner'
    )
    if df_comparacion.empty:
        print("Error: No se pudieron unir los resultados."); sys.exit(1)
    print("\nPrimeras filas de la comparación:")
    print(df_comparacion[['Nombre Empresa', 'Nivel Economico (Reglas)', 'Nivel Economico (Predicho Modelo)']].head())
    reglas_labels = df_comparacion['Nivel Economico (Reglas)'].astype(str)
    modelo_labels = df_comparacion['Nivel Economico (Predicho Modelo)'].astype(str)
    all_labels = sorted(list(set(reglas_labels) | set(modelo_labels)))
    print("\nReporte de Clasificación (Modelo Optimizado vs. Reglas en Nuevos Datos):")
    accuracy = accuracy_score(reglas_labels, modelo_labels)
    print(f"Accuracy Modelo Optimizado vs Reglas: {accuracy:.4f}")
    print(classification_report(reglas_labels, modelo_labels, labels=all_labels, zero_division=0))

    # 8. Graficar
    print("\nGenerando gráfico de Matriz de Confusión...")
    graficar_matriz_confusion(reglas_labels, modelo_labels, labels=all_labels,
                              title='Matriz de Confusión - Modelo Optimizado vs. Reglas (Nuevos Datos)')


    # --- Ejemplo de cómo cargar y usar el modelo guardado ---
    print("\n\n--- DEMO: Cargando y usando el modelo guardado ---")
    # Supongamos que tenemos otro DataFrame nuevo (podríamos generar otro o reusar df_nuevos_generados)
    df_para_demo = df_nuevos_generados.sample(5) # Tomar 5 muestras aleatorias
    print("Nuevos datos para la demo (primeras filas):")
    print(df_para_demo[BASE_FEATURES].head())

    predicciones_cargadas = cargar_y_usar_modelo_guardado(
        df_para_demo,
        BASE_FEATURES, # Pasamos las features esperadas
        bundle_path=BUNDLE_PATH # Usamos la ruta donde guardamos el bundle
    )

    if predicciones_cargadas is not None:
        print("\nPredicciones realizadas con el modelo CARGADO:")
        df_resultado_demo = df_para_demo[['Nombre Empresa'] + BASE_FEATURES].copy()
        df_resultado_demo['Nivel Predicho (Cargado)'] = predicciones_cargadas
        print(df_resultado_demo)
    else:
        print("\nNo se pudieron obtener predicciones con el modelo cargado.")


    print("\n--- Proceso completado ---")