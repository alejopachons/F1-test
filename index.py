import streamlit as st
import pandas as pd
import os
import fastf1
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np

st.set_page_config(page_title="F1 2026 Predictor", layout="wide")
st.title("🏎️ F1 Podium & Position Predictor: Era 2022-2026")

BASE_DATA_FILE = "f1_dataset_2022_2025.csv"
ANALYZED_DATA_FILE = "f1_analyzed_data.csv"

# --- LISTADO OFICIAL 2026 ---
# Mapeo de abreviaturas oficiales de F1 (3 letras) a los nombres que proporcionaste
PILOTOS_2026 = {
    'ALB': 'Albon', 'ALO': 'Alonso', 'ANT': 'Antonelli', 'BEA': 'Bearman',
    'BOR': 'Bortoleto', 'BOT': 'Bottas', 'COL': 'Colapinto', 'GAS': 'Gasly',
    'HAD': 'Hadjar', 'HAM': 'Hamilton', 'HUL': 'Hülkenberg', 'LAW': 'Lawson',
    'LEC': 'Leclerc', 'LIN': 'Lindblad', 'NOR': 'Norris', 'OCO': 'Ocon',
    'PER': 'Perez', 'PIA': 'Piastri', 'RUS': 'Russell', 'SAI': 'Sainz',
    'STR': 'Stroll', 'VER': 'Verstappen'
}

@st.cache_data
def load_base_data():
    if os.path.exists(BASE_DATA_FILE):
        return pd.read_csv(BASE_DATA_FILE)
    else:
        st.error(f"⚠️ No se encontró {BASE_DATA_FILE}.")
        return pd.DataFrame()

@st.cache_data
def fetch_2026_data():
    all_results = []
    for round_num in range(1, 25): 
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
            break 
    return pd.DataFrame(all_results)

df_base = load_base_data()
df_2026 = fetch_2026_data()

if not df_base.empty:
    if not df_2026.empty:
        df = pd.concat([df_base, df_2026], ignore_index=True)
    else:
        df = df_base.copy()

    df = df.dropna(subset=['finish', 'grid'])
    # Convertimos finish a numérico para el regresor
    df['finish'] = pd.to_numeric(df['finish'], errors='coerce')
    df = df.dropna(subset=['finish'])
    
    df['is_podium'] = df['finish'].apply(lambda x: 1 if x <= 3 else 0)
    
    # --- LÓGICA DE ROOKIES ---
    pilotos_base = df_base['pilot'].unique() if not df_base.empty else []
    # Usamos las keys (abreviaturas) de tu lista
    rookies_list = [p for p in PILOTOS_2026.keys() if p not in pilotos_base]

    # --- ENCODERS (Asegurando que todos los de 2026 existan) ---
    le_pilot = LabelEncoder()
    le_circuit = LabelEncoder()
    
    # Inyectamos todos los pilotos 2026 al encoder para que no falle con los rookies sin datos
    todos_los_pilotos = pd.concat([df['pilot'], pd.Series(list(PILOTOS_2026.keys()))]).unique()
    le_pilot.fit(todos_los_pilotos)
    
    df['pilot_id'] = le_pilot.transform(df['pilot'].astype(str))
    df['circuit_id'] = le_circuit.fit_transform(df['circuit'].astype(str))
    
    X = df[['grid', 'pilot_id', 'circuit_id', 'year']]
    y_class = df['is_podium'] # Para probabilidad de podio
    y_reg = df['finish']      # Para predecir posición final

    # --- CONFIGURACIÓN EN SIDEBAR ---
    st.sidebar.header("⚙️ Configuración del Modelo")
    model_choice = st.sidebar.selectbox("Elige el modelo estadístico", ["Random Forest", "Gradient Boosting", "Lineal / Logística"])
    
    # --- INDICADOR DE DATOS ACTUALIZADOS ---
    st.sidebar.markdown("---")
    st.sidebar.subheader("📅 Última actualización")
    if not df_2026.empty:
        last_race = df_2026.iloc[-1]['circuit']
        st.sidebar.success(f"**Datos al día.**\n\nÚltima carrera procesada:\n🏁 {last_race} (2026)")
    else:
        last_year = df_base['year'].max()
        last_race = df_base[df_base['year'] == last_year].iloc[-1]['circuit']
        st.sidebar.warning(f"**Esperando inicio 2026.**\n\nÚltima carrera en base:\n🏁 {last_race} ({last_year})")

    # --- ENTRENAMIENTO DE DOS MODELOS (Clasificador y Regresor) ---
    if model_choice == "Random Forest":
        model_class = RandomForestClassifier(n_estimators=100, random_state=42)
        model_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_choice == "Gradient Boosting":
        model_class = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model_reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
    else:
        model_class = LogisticRegression(max_iter=1000)
        model_reg = LinearRegression()
        
    model_class.fit(X, y_class)
    model_reg.fit(X, y_reg)

    # --- PREDICCIONES 2026 ---
    st.subheader("📊 Predicción Oficial Parrilla 2026")
    selected_circuit = st.selectbox("Selecciona el Circuito a simular", df['circuit'].unique())
    c_id = le_circuit.transform([selected_circuit])[0]
    
    predictions = []
    
    # Iteramos sobre TU lista oficial de 2026
    for pilot_abbr, pilot_name in PILOTOS_2026.items():
        p_id = le_pilot.transform([pilot_abbr])[0]
        
        # Calcular grid promedio (Si es rookie, asumimos posición 15 para empezar)
        historial_piloto = df[df['pilot'] == pilot_abbr]['grid']
        avg_grid = historial_piloto.mean() if not historial_piloto.empty else 15.0
        
        # Predicción de Probabilidad de Podio
        prob = model_class.predict_proba([[avg_grid, p_id, c_id, 2026]])[0][1]
        
        # Predicción de Posición Final
        pos_pred = model_reg.predict([[avg_grid, p_id, c_id, 2026]])[0]
        # Limitamos la predicción entre el puesto 1 y el 20
        pos_pred = max(1, min(20, round(pos_pred))) 
        
        nombre_display = f"{pilot_name} [Rookie 🔰]" if pilot_abbr in rookies_list else pilot_name
        
        predictions.append({
            "Piloto": nombre_display, 
            "Posición Salida Prom.": round(avg_grid, 1), 
            "Posición Final Prevista": int(pos_pred),
            "Probabilidad de Podio (%)": round(prob * 100, 2)
        })

    # Mostramos la tabla ordenada por la posición final prevista (del 1ro al último)
    df_preds = pd.DataFrame(predictions).sort_values(by=["Posición Final Prevista", "Probabilidad de Podio (%)"], ascending=[True, False])
    # Reseteamos el índice para que parezca la posición del campeonato
    df_preds.index = np.arange(1, len(df_preds) + 1) 
    
    st.dataframe(df_preds, use_container_width=True)

    # Guardado silencioso del CSV
    df.to_csv(ANALYZED_DATA_FILE, index=False)

st.markdown("---")
if st.button("🔄 Buscar nuevos resultados de 2026 en FastF1"):
    st.cache_data.clear()
    st.rerun()
