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
    
    pilotos_base = df_base['pilot'].unique() if not df_base.empty else []
    rookies_list = [p for p in PILOTOS_2026.keys() if p not in pilotos_base]

    le_pilot = LabelEncoder()
    le_circuit = LabelEncoder()
    
    todos_los_pilotos = pd.concat([df['pilot'], pd.Series(list(PILOTOS_2026.keys()))]).unique()
    le_pilot.fit(todos_los_pilotos)
    
    todos_los_circuitos = pd.concat([df['circuit'], pd.Series(CALENDARIO_2026)]).unique()
    le_circuit.fit(todos_los_circuitos)
    
    df['pilot_id'] = le_pilot.transform(df['pilot'].astype(str))
    df['circuit_id'] = le_circuit.transform(df['circuit'].astype(str))
    
    X = df[['grid', 'pilot_id', 'circuit_id', 'year']]
    y_class = df['is_podium'] 
    y_reg = df['finish']      

    # --- ENTRENAMIENTO DE LOS 3 MODELOS SIMULTÁNEAMENTE ---
    modelos_class = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Lineal / Logística": LogisticRegression(max_iter=1000)
    }
    
    modelos_reg = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Lineal / Logística": LinearRegression()
    }

    # Entrenamos todos
    for name in modelos_class.keys():
        modelos_class[name].fit(X, y_class)
        modelos_reg[name].fit(X, y_reg)

    # --- SIDEBAR: ESTATUS Y SELECTOR ---
    st.sidebar.subheader("📅 Última actualización")
    if not df_2026.empty:
        last_race = df_2026.iloc[-1]['circuit']
        st.sidebar.success(f"**Datos al día.**\n🏁 {last_race} (2026)")
    else:
        last_year = df_base['year'].max()
        try:
            last_race = df_base[df_base['year'] == last_year].iloc[-1]['circuit']
        except IndexError:
            last_race = "Desconocida"
        st.sidebar.info(f"**Esperando inicio 2026.**\n🏁 {last_race} ({last_year})")

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Configuración de Visualización")
    modelo_seleccionado = st.sidebar.radio("Selecciona el modelo a visualizar:", 
                                         ["Random Forest", "Gradient Boosting", "Lineal / Logística"])
    
    selected_circuit = st.sidebar.selectbox("Próxima carrera", CALENDARIO_2026)
    c_id = le_circuit.transform([selected_circuit])[0]

    # --- GENERAR PREDICCIONES SEGÚN EL MODELO SELECCIONADO ---
    st.subheader(f"📊 Predicción Parrilla 2026 - {selected_circuit}")
    st.markdown(f"**Modelo activo:** {modelo_seleccionado}")
    
    predictions = []
    
    for pilot_abbr, info in PILOTOS_2026.items():
        p_id = le_pilot.transform([pilot_abbr])[0]
        
        historial_piloto = df[df['pilot'] == pilot_abbr]['grid']
        avg_grid = historial_piloto.mean() if not historial_piloto.empty else 15.0
        
        # Usar el modelo seleccionado
        prob = modelos_class[modelo_seleccionado].predict_proba([[avg_grid, p_id, c_id, 2026]])[0][1]
        pos_pred = modelos_reg[modelo_seleccionado].predict([[avg_grid, p_id, c_id, 2026]])[0]
        
        nombre_display = f"{info['nombre']} 🔰" if pilot_abbr in rookies_list else info['nombre']
        
        predictions.append({
            "Piloto": nombre_display, 
            "Posición Final Prevista": pos_pred, # Lo mantenemos como float temporalmente para el ordenamiento
            "Prob. Podio (%)": round(prob * 100, 2)
        })

    # --- ORDENAMIENTO ESTRICTO Y VISUALIZACIÓN ---
    # Ordenamos por la predicción cruda (float) para desempatar y luego por probabilidad
    df_preds = pd.DataFrame(predictions).sort_values(by=["Posición Final Prevista", "Prob. Podio (%)"], ascending=[True, False])
    
    # Asignamos posiciones del 1 al 22 estrictamente (índice)
    df_preds.index = np.arange(1, len(df_preds) + 1)
    
    # Ahora que ya está ordenado, redondeamos visualmente la "Posición Final Prevista" a entero y limitamos a 22
    df_preds['Posición Final Prevista'] = df_preds.index
    
    # Mostramos la tabla limpia
    st.dataframe(df_preds[['Piloto', 'Posición Final Prevista', 'Prob. Podio (%)']], use_container_width=True)

    # --- GRÁFICO DE IMPORTANCIA DE VARIABLES ---
    st.markdown("---")
    st.subheader(f"🔍 ¿Qué miró el modelo '{modelo_seleccionado}' para decidir?")
    
    # Obtener importancias dependiendo del tipo de modelo (usamos el regresor para el análisis general)
    modelo_actual = modelos_reg[modelo_seleccionado]
    
    if modelo_seleccionado in ["Random Forest", "Gradient Boosting"]:
        importances = modelo_actual.feature_importances_
    else: # Regresión Lineal
        # Tomamos el valor absoluto de los coeficientes para ver su impacto total
        importances = np.abs(modelo_actual.coef_)
        # Normalizamos para que sume 100% y sea comparable visualmente
        importances = importances / np.sum(importances)

    # Crear el gráfico
    fig, ax = plt.subplots(figsize=(8, 4))
    variables = ['Posición de Salida (Grid)', 'Piloto', 'Circuito', 'Año']
    
    # Colores F1
    colors = ['#FF1801' if val == max(importances) else '#808080' for val in importances]
    
    ax.barh(variables, importances, color=colors)
    ax.set_xlabel("Impacto en la predicción")
    ax.set_title(f"Peso de las variables en {modelo_seleccionado}")
    
    # Invertir eje Y para que la más importante salga arriba
    ax.invert_yaxis() 
    
    st.pyplot(fig)

    df.to_csv(ANALYZED_DATA_FILE, index=False)

st.markdown("---")
if st.button("🔄 Buscar nuevos resultados de 2026 en FastF1"):
    st.cache_data.clear()
    st.rerun()
