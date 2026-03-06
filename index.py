import streamlit as st
import pandas as pd
import os
import fastf1
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="F1 2026 Predictor", layout="wide")
st.title("🏎️ F1 Podium Predictor: Era 2022-2026")
st.markdown("Esta app entrena el modelo a partir del histórico local y busca nuevos datos de 2026 en segundo plano.")

BASE_DATA_FILE = "f1_dataset_2022_2025.csv"
ANALYZED_DATA_FILE = "f1_analyzed_data.csv"

# --- 1. CARGA DEL DATASET HISTÓRICO (2022-2025) ---
@st.cache_data
def load_base_data():
    if os.path.exists(BASE_DATA_FILE):
        return pd.read_csv(BASE_DATA_FILE)
    else:
        st.error(f"⚠️ No se encontró el archivo base: {BASE_DATA_FILE}. Por favor, asegúrate de que esté en la misma carpeta.")
        return pd.DataFrame()

# --- 2. EXTRACCIÓN DE DATOS NUEVOS (2026) ---
@st.cache_data
def fetch_2026_data():
    all_results = []
    # Busca las primeras 10 rondas de 2026 (ajusta según avance el calendario)
    for round_num in range(1, 11): 
        try:
            session = fastf1.get_session(2026, round_num, 'R')
            session.load(telemetry=False, weather=False, messages=False)
            for _, row in session.results.iterrows():
                all_results.append({
                    'year': 2026, 
                    'pilot': row['Abbreviation'],
                    'grid': row['GridPosition'], 
                    'finish': row['Position'],
                    'circuit': session.event['EventName']
                })
        except Exception:
            # Si la carrera aún no ha ocurrido, la API da error y saltamos a la siguiente
            continue
    return pd.DataFrame(all_results)

# Cargamos ambos sets de datos
df_base = load_base_data()
df_2026 = fetch_2026_data()

if not df_base.empty:
    # --- 3. COMBINAR DATOS Y PREPARAR ---
    # Unimos el histórico con lo nuevo que se haya corrido en 2026
    if not df_2026.empty:
        df = pd.concat([df_base, df_2026], ignore_index=True)
    else:
        df = df_base.copy()

    # Limpieza
    df = df.dropna(subset=['finish', 'grid'])
    df['is_podium'] = df['finish'].apply(lambda x: 1 if pd.to_numeric(x, errors='coerce') <= 3 else 0)
    
    # Encoders
    le_pilot = LabelEncoder()
    le_circuit = LabelEncoder()
    df['pilot_id'] = le_pilot.fit_transform(df['pilot'].astype(str))
    df['circuit_id'] = le_circuit.fit_transform(df['circuit'].astype(str))
    
    X = df[['grid', 'pilot_id', 'circuit_id', 'year']]
    y = df['is_podium']

    # --- 4. CONFIGURACIÓN Y ENTRENAMIENTO DEL MODELO ---
    st.sidebar.header("Configuración del Modelo")
    model_choice = st.sidebar.selectbox("Elige el modelo estadístico", 
                                        ["Random Forest", "Gradient Boosting", "Regresión Logística"])
    
    if model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_choice == "Gradient Boosting":
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    else:
        model = LogisticRegression(max_iter=1000)
        
    # Entrenamiento con el CSV unificado
    model.fit(X, y)

    # --- 5. IMPORTANCIA DE VARIABLES VISUAL ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Importancia de las variables")
    
    if model_choice in ["Random Forest", "Gradient Boosting"]:
        importances = model.feature_importances_
    else:
        importances = abs(model.coef_[0])
        
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(X.columns, importances, color='#ff1801') # Color rojo F1
    ax.set_xlabel("Peso en la decisión")
    st.sidebar.pyplot(fig)

    # --- 6. PREDICCIONES 2026 ---
    st.subheader("📊 Probabilidades de Podio")
    
    selected_circuit = st.selectbox("Selecciona el Circuito a simular", df['circuit'].unique())
    c_id = le_circuit.transform([selected_circuit])[0]
    
    latest_year = df['year'].max()
    active_pilots = df[df['year'] == latest_year]['pilot'].unique()
    
    predictions = []
    for pilot in active_pilots:
        try:
            p_id = le_pilot.transform([pilot])[0]
            avg_grid = df[df['pilot'] == pilot]['grid'].mean()
            if pd.isna(avg_grid): avg_grid = 10 
            
            prob = model.predict_proba([[avg_grid, p_id, c_id, 2026]])[0][1]
            predictions.append({
                "Piloto": pilot, 
                "Posición Salida Promedio": round(avg_grid, 1), 
                "Probabilidad de Podio (%)": round(prob * 100, 2)
            })
        except ValueError:
            continue

    df_preds = pd.DataFrame(predictions).sort_values(by="Probabilidad de Podio (%)", ascending=False)
    st.dataframe(df_preds, hide_index=True, use_container_width=True)

    # --- 7. MOSTRAR DATOS HISTÓRICOS (SIN 2025) ---
    st.markdown("---")
    st.subheader("Últimos registros en la base de datos (Excluyendo 2025)")
    df_display = df[df['year'] != 2025]
    st.dataframe(df_display.tail(10), use_container_width=True)

    # --- 8. GUARDADO EN SEGUNDO PLANO ---
    # Esto ocurre de forma invisible y no entorpece la interfaz de Streamlit
    # Guarda el dataset limpio, encodificado y con los datos combinados de 2026.
    df.to_csv(ANALYZED_DATA_FILE, index=False)

# Botón para forzar actualización de la caché si se corrió una nueva carrera de 2026
st.markdown("---")
if st.button("🔄 Buscar nuevos resultados de 2026 en FastF1"):
    st.cache_data.clear()
    st.rerun()
