import streamlit as st
import pandas as pd
import os
import fastf1
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="F1 2026 Predictor", layout="wide")
st.title("🏎️ F1 Podium Predictor: Era 2022-2026")

BASE_DATA_FILE = "f1_dataset_2022_2025.csv"
ANALYZED_DATA_FILE = "f1_analyzed_data.csv"

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
    # Busca dinámicamente las carreras de 2026 que ya se hayan corrido
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
            break # Si falla, asumimos que llegamos a una carrera futura y detenemos la búsqueda
    return pd.DataFrame(all_results)

df_base = load_base_data()
df_2026 = fetch_2026_data()

if not df_base.empty:
    if not df_2026.empty:
        df = pd.concat([df_base, df_2026], ignore_index=True)
    else:
        df = df_base.copy()

    df = df.dropna(subset=['finish', 'grid'])
    df['is_podium'] = df['finish'].apply(lambda x: 1 if pd.to_numeric(x, errors='coerce') <= 3 else 0)
    
    # --- LÓGICA DE ROOKIES ---
    pilotos_base = df_base['pilot'].unique() if not df_base.empty else []
    df['is_rookie'] = df.apply(lambda row: 1 if row['year'] == 2026 and row['pilot'] not in pilotos_base else 0, axis=1)
    rookies_list = df[df['is_rookie'] == 1]['pilot'].unique()

    # Encoders entrenados con toda la base (incluyendo rookies para que no haya errores)
    le_pilot = LabelEncoder()
    le_circuit = LabelEncoder()
    df['pilot_id'] = le_pilot.fit_transform(df['pilot'].astype(str))
    df['circuit_id'] = le_circuit.fit_transform(df['circuit'].astype(str))
    
    X = df[['grid', 'pilot_id', 'circuit_id', 'year']]
    y = df['is_podium']

    st.sidebar.header("Configuración del Modelo")
    model_choice = st.sidebar.selectbox("Elige el modelo estadístico", ["Random Forest", "Gradient Boosting", "Regresión Logística"])
    
    if model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_choice == "Gradient Boosting":
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    else:
        model = LogisticRegression(max_iter=1000)
        
    model.fit(X, y)

    # --- PREDICCIONES Y ETIQUETA ROOKIE ---
    st.subheader("📊 Probabilidades de Podio 2026")
    selected_circuit = st.selectbox("Selecciona el Circuito a simular", df['circuit'].unique())
    c_id = le_circuit.transform([selected_circuit])[0]
    
    active_pilots = df[df['year'] == 2026]['pilot'].unique()
    if len(active_pilots) == 0: # Por si no hay datos de 2026 aún
        active_pilots = df[df['year'] == 2025]['pilot'].unique()

    predictions = []
    for pilot in active_pilots:
        p_id = le_pilot.transform([pilot])[0]
        
        # Calcular grid promedio. Si es rookie sin carreras, le asignamos 15 por defecto.
        historial_piloto = df[df['pilot'] == pilot]['grid']
        avg_grid = historial_piloto.mean() if not historial_piloto.empty else 15.0
        
        prob = model.predict_proba([[avg_grid, p_id, c_id, 2026]])[0][1]
        
        # Etiqueta de Rookie
        nombre_display = f"{pilot} [Rookie 🔰]" if pilot in rookies_list else pilot
        
        predictions.append({
            "Piloto": nombre_display, 
            "Posición Salida Promedio": round(avg_grid, 1), 
            "Probabilidad de Podio (%)": round(prob * 100, 2)
        })

    df_preds = pd.DataFrame(predictions).sort_values(by="Probabilidad de Podio (%)", ascending=False)
    st.dataframe(df_preds, hide_index=True, use_container_width=True)

    df.to_csv(ANALYZED_DATA_FILE, index=False)

st.markdown("---")
if st.button("🔄 Buscar nuevos resultados de 2026 en FastF1"):
    st.cache_data.clear()
    st.rerun()
