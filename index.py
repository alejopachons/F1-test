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
st.markdown("Esta aplicación predice las probabilidades de podio utilizando datos históricos y modelos estadísticos.")

DATA_FILE = "f1_historical_data.csv"

# --- 1. DESCARGA Y GUARDADO LOCAL ---
@st.cache_data
def load_or_fetch_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    else:
        st.warning("Archivo CSV no encontrado. Descargando historial por única vez (esto tomará unos minutos)...")
        all_results = []
        # Incluimos 2025 para que la base de datos descargable esté completa
        for year in [2022, 2023, 2024, 2025, 2026]:
            for round_num in range(1, 6): # Ajusta este número para traer más carreras de la temporada
                try:
                    session = fastf1.get_session(year, round_num, 'R')
                    session.load(telemetry=False, weather=False, messages=False)
                    for _, row in session.results.iterrows():
                        all_results.append({
                            'year': year, 
                            'pilot': row['Abbreviation'],
                            'grid': row['GridPosition'], 
                            'finish': row['Position'],
                            'circuit': session.event['EventName']
                        })
                except Exception as e:
                    continue
        df_new = pd.DataFrame(all_results)
        df_new.to_csv(DATA_FILE, index=False)
        return df_new

df = load_or_fetch_data()

if not df.empty:
    # --- 2. PREPARACIÓN DE DATOS Y ENCODERS ---
    # Limpieza básica
    df = df.dropna(subset=['finish', 'grid'])
    # Convertimos la posición final a número y definimos podio (1, 2, o 3)
    df['is_podium'] = df['finish'].apply(lambda x: 1 if pd.to_numeric(x, errors='coerce') <= 3 else 0)
    
    # Encoders para convertir texto a números
    le_pilot = LabelEncoder()
    le_circuit = LabelEncoder()
    
    df['pilot_id'] = le_pilot.fit_transform(df['pilot'].astype(str))
    df['circuit_id'] = le_circuit.fit_transform(df['circuit'].astype(str))
    
    X = df[['grid', 'pilot_id', 'circuit_id', 'year']]
    y = df['is_podium']

    # --- 3. SELECCIÓN DE MODELO ---
    st.sidebar.header("Configuración del Modelo")
    model_choice = st.sidebar.selectbox("Elige el modelo estadístico", 
                                        ["Random Forest", "Gradient Boosting", "Regresión Logística"])
    
    if model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_choice == "Gradient Boosting":
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    else:
        model = LogisticRegression(max_iter=1000)
        
    model.fit(X, y)

    # --- 4. IMPORTANCIA DE VARIABLES ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("Importancia de las variables")
    
    if model_choice in ["Random Forest", "Gradient Boosting"]:
        importances = model.feature_importances_
    else:
        importances = abs(model.coef_[0])
        
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(X.columns, importances, color='skyblue')
    ax.set_xlabel("Peso en la decisión")
    st.sidebar.pyplot(fig)

    # --- 5. PREDICCIÓN PARA TODOS LOS PILOTOS ---
    st.subheader("📊 Probabilidades de Podio")
    
    selected_circuit = st.selectbox("Selecciona el Circuito a simular", df['circuit'].unique())
    c_id = le_circuit.transform([selected_circuit])[0]
    
    latest_year = df['year'].max()
    active_pilots = df[df['year'] == latest_year]['pilot'].unique()
    
    predictions = []
    for pilot in active_pilots:
        try:
            p_id = le_pilot.transform([pilot])[0]
            # Usamos la posición promedio de salida histórica del piloto para la simulación
            avg_grid = df[df['pilot'] == pilot]['grid'].mean()
            if pd.isna(avg_grid): avg_grid = 10 
            
            prob = model.predict_proba([[avg_grid, p_id, c_id, 2026]])[0][1]
            predictions.append({"Piloto": pilot, "Posición Salida Promedio": round(avg_grid, 1), "Probabilidad de Podio (%)": round(prob * 100, 2)})
        except ValueError:
            continue

    df_preds = pd.DataFrame(predictions).sort_values(by="Probabilidad de Podio (%)", ascending=False)
    st.dataframe(df_preds, hide_index=True, use_container_width=True)

    # --- 6. DATOS HISTÓRICOS (SIN 2025 EN LA INTERFAZ) ---
    st.markdown("---")
    st.subheader("Muestra de Datos Históricos (Ocultando 2025)")
    df_display = df[df['year'] != 2025]
    st.dataframe(df_display.tail(10), use_container_width=True)

    # --- 7. BOTÓN DE DESCARGA CSV ---
    st.markdown("---")
    st.subheader("📥 Exportar Datos")
    
    csv_data = df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="Descargar Base de Datos Completa (CSV)",
        data=csv_data,
        file_name="f1_dataset_2022_2026.csv",
        mime="text/csv",
        help="Descarga el historial completo usado para entrenar el modelo (incluye 2025)."
    )

else:
    st.error("No hay datos disponibles.")

# --- 8. BOTÓN PARA FORZAR ACTUALIZACIÓN ---
st.markdown("---")
if st.button("🔄 Borrar caché y descargar nuevos resultados desde FastF1"):
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
    st.cache_data.clear()
    st.rerun()
