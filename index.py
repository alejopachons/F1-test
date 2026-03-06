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

# --- 1. ALINEACIÓN OFICIAL 2026 ---
# Mapeo de abreviatura oficial -> Nombre del piloto y su Equipo
PILOTOS_2026 = {
    'GAS': {'nombre': 'Gasly', 'equipo': 'Alpine'},
    'COL': {'nombre': 'Colapinto', 'equipo': 'Alpine'},
    'ALO': {'nombre': 'Alonso', 'equipo': 'Aston Martin'},
    'STR': {'nombre': 'Stroll', 'equipo': 'Aston Martin'},
    'ALB': {'nombre': 'Albon', 'equipo': 'Williams'},
    'SAI': {'nombre': 'Sainz', 'equipo': 'Williams'},
    'BOR': {'nombre': 'Bortoleto', 'equipo': 'Audi'},
    'HUL': {'nombre': 'Hülkenberg', 'equipo': 'Audi'},
    'PER': {'nombre': 'Pérez', 'equipo': 'Cadillac'},
    'BOT': {'nombre': 'Bottas', 'equipo': 'Cadillac'},
    'LEC': {'nombre': 'Leclerc', 'equipo': 'Ferrari'},
    'HAM': {'nombre': 'Hamilton', 'equipo': 'Ferrari'},
    'OCO': {'nombre': 'Ocon', 'equipo': 'Haas'},
    'BEA': {'nombre': 'Bearman', 'equipo': 'Haas'},
    'NOR': {'nombre': 'Norris', 'equipo': 'McLaren'},
    'PIA': {'nombre': 'Piastri', 'equipo': 'McLaren'},
    'ANT': {'nombre': 'Antonelli', 'equipo': 'Mercedes'},
    'RUS': {'nombre': 'Russell', 'equipo': 'Mercedes'},
    'LAW': {'nombre': 'Lawson', 'equipo': 'Racing Bulls'},
    'LIN': {'nombre': 'Lindblad', 'equipo': 'Racing Bulls'},
    'VER': {'nombre': 'Verstappen', 'equipo': 'Red Bull'},
    'HAD': {'nombre': 'Hadjar', 'equipo': 'Red Bull'}
}

# --- 2. CALENDARIO OFICIAL 2026 ---
CALENDARIO_2026 = [
    "Albert Park Circuit", "Shanghai International Circuit", "Suzuka Circuit",
    "Bahrain International Circuit", "Jeddah Corniche Circuit", "Miami International Autodrome",
    "Circuit Gilles Villeneuve", "Circuit de Monaco", "Circuit de Barcelona-Catalunya",
    "Red Bull Ring", "Silverstone Circuit", "Circuit de Spa-Francorchamps",
    "Hungaroring", "Circuit Zandvoort", "Monza Circuit", "Madring", 
    "Baku City Circuit", "Marina Bay Street Circuit", "Circuit of the Americas",
    "Autódromo Hermanos Rodríguez", "Interlagos Circuit", "Las Vegas Strip Circuit",
    "Lusail International Circuit", "Yas Marina Circuit"
]

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
    # Busca dinámicamente las carreras de 2026
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
    df['finish'] = pd.to_numeric(df['finish'], errors='coerce')
    df = df.dropna(subset=['finish'])
    df['is_podium'] = df['finish'].apply(lambda x: 1 if x <= 3 else 0)
    
    # --- IDENTIFICACIÓN DE ROOKIES ---
    pilotos_base = df_base['pilot'].unique() if not df_base.empty else []
    rookies_list = [p for p in PILOTOS_2026.keys() if p not in pilotos_base]

    # --- ENCODERS ---
    le_pilot = LabelEncoder()
    le_circuit = LabelEncoder()
    
    todos_los_pilotos = pd.concat([df['pilot'], pd.Series(list(PILOTOS_2026.keys()))]).unique()
    le_pilot.fit(todos_los_pilotos)
    
    # Aseguramos que el nuevo circuito "Madring" esté en el encoder aunque no haya historial
    todos_los_circuitos = pd.concat([df['circuit'], pd.Series(CALENDARIO_2026)]).unique()
    le_circuit.fit(todos_los_circuitos)
    
    df['pilot_id'] = le_pilot.transform(df['pilot'].astype(str))
    df['circuit_id'] = le_circuit.transform(df['circuit'].astype(str))
    
    X = df[['grid', 'pilot_id', 'circuit_id', 'year']]
    y_class = df['is_podium'] 
    y_reg = df['finish']      

    # --- SIDEBAR: CONFIGURACIÓN Y ESTATUS ---
    st.sidebar.header("⚙️ Configuración del Modelo")
    model_choice = st.sidebar.selectbox("Elige el modelo estadístico", ["Random Forest", "Gradient Boosting", "Lineal / Logística"])
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📅 Última actualización")
    if not df_2026.empty:
        last_race = df_2026.iloc[-1]['circuit']
        st.sidebar.success(f"**Datos al día.**\n\nÚltima carrera procesada:\n🏁 {last_race} (2026)")
    else:
        last_year = df_base['year'].max()
        # Manejo de error en caso de que el max year de la base no coincida
        try:
            last_race = df_base[df_base['year'] == last_year].iloc[-1]['circuit']
        except IndexError:
            last_race = "Desconocida"
        st.sidebar.info(f"**Esperando inicio 2026.**\n\nEl GP de Australia aún no se corre o no hay datos cargados.\n\nÚltima carrera en base:\n🏁 {last_race} ({last_year})")

    # --- ENTRENAMIENTO ---
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
    
    # Usamos el calendario oficial para el dropdown
    selected_circuit = st.selectbox("Selecciona la próxima carrera (Calendario 2026)", CALENDARIO_2026)
    c_id = le_circuit.transform([selected_circuit])[0]
    
    predictions = []
    
    for pilot_abbr, info in PILOTOS_2026.items():
        p_id = le_pilot.transform([pilot_abbr])[0]
        
        historial_piloto = df[df['pilot'] == pilot_abbr]['grid']
        avg_grid = historial_piloto.mean() if not historial_piloto.empty else 15.0
        
        prob = model_class.predict_proba([[avg_grid, p_id, c_id, 2026]])[0][1]
        
        pos_pred = model_reg.predict([[avg_grid, p_id, c_id, 2026]])[0]
        pos_pred = max(1, min(20, round(pos_pred))) 
        
        nombre_display = f"{info['nombre']} 🔰" if pilot_abbr in rookies_list else info['nombre']
        
        predictions.append({
            "Piloto": nombre_display, 
            "Equipo": info['equipo'],
            "Posición Salida Prom.": round(avg_grid, 1), 
            "Posición Final Prevista": int(pos_pred),
            "Prob. Podio (%)": round(prob * 100, 2)
        })

    df_preds = pd.DataFrame(predictions).sort_values(by=["Posición Final Prevista", "Prob. Podio (%)"], ascending=[True, False])
    df_preds.index = np.arange(1, len(df_preds) + 1) 
    
    st.dataframe(df_preds, use_container_width=True)

    df.to_csv(ANALYZED_DATA_FILE, index=False)

st.markdown("---")
if st.button("🔄 Buscar nuevos resultados de 2026 en FastF1"):
    st.cache_data.clear()
    st.rerun()
