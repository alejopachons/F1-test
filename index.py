import streamlit as st
import fastf1
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import datetime

# Configuración de la página
st.set_page_config(page_title="F1 2026 Predictor", layout="wide")

st.title("🏎️ F1 Podium Predictor: Era 2022-2026")
st.markdown("""
Esta app extrae datos en tiempo real de la API de FastF1 y entrena un modelo 
conforme avanza la temporada 2026.
""")

# --- PASO 1: FUNCIÓN PARA EXTRAER DATOS ---
@st.cache_data # Para no saturar la API y que la app sea rápida
def get_f1_data(years):
    all_results = []
    for year in years:
        # Intentamos obtener las primeras 5 carreras de cada año para el ejemplo
        # En una app real, podrías iterar por todo el calendario
        for round_num in range(1, 6): 
            try:
                session = fastf1.get_session(year, round_num, 'R')
                session.load(telemetry=False, weather=False, messages=False)
                results = session.results
                
                for _, row in results.iterrows():
                    all_results.append({
                        'year': year,
                        'pilot': row['Abbreviation'],
                        'team': row['TeamName'],
                        'grid': row['GridPosition'],
                        'finish': row['Position'],
                        'circuit': session.event['EventName']
                    })
            except:
                continue # Si la carrera aún no ha ocurrido
    return pd.DataFrame(all_results)

# --- PASO 2: CARGA DE DATOS ---
with st.spinner("Extrayendo datos históricos y de 2026..."):
    # Incluimos desde 2022 hasta el año actual 2026
    years_to_load = [2022, 2023, 2024, 2025, 2026]
    df = get_f1_data(years_to_load)

if not df.empty:
    # --- PASO 3: PREPARACIÓN DEL MODELO ---
    # Creamos la variable objetivo: 1 si quedó en el podio (1, 2, 3), 0 si no.
    df['is_podium'] = df['finish'].apply(lambda x: 1 if x <= 3 else 0)
    
    # Convertimos textos a números para el modelo (Encoding)
    df['pilot_id'] = df['pilot'].astype('category').cat.codes
    df['circuit_id'] = df['circuit'].astype('category').cat.codes
    
    # Entrenamos el modelo
    X = df[['grid', 'pilot_id', 'circuit_id', 'year']]
    y = df['is_podium']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # --- PASO 4: INTERFAZ DE PREDICCIÓN ---
    st.sidebar.header("Parámetros de Predicción")
    
    selected_pilot = st.sidebar.selectbox("Selecciona Piloto", df['pilot'].unique())
    selected_circuit = st.sidebar.selectbox("Circuito", df['circuit'].unique())
    grid_pos = st.sidebar.slider("Posición de salida (Grid)", 1, 20, 1)
    
    # Ejecutar predicción
    p_id = pd.Categorical(df['pilot']).categories.get_loc(selected_pilot)
    c_id = pd.Categorical(df['circuit']).categories.get_loc(selected_circuit)
    
    prediction_proba = model.predict_proba([[grid_pos, p_id, c_id, 2026]])[0][1]

    # --- PASO 5: MOSTRAR RESULTADOS ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label=f"Probabilidad de Podio para {selected_pilot}", 
                  value=f"{round(prediction_proba * 100, 2)}%")
        
        st.progress(prediction_proba)
        
    with col2:
        st.write("### Últimos resultados cargados")
        st.dataframe(df.tail(10))

else:
    st.error("No se pudieron cargar los datos. Verifica tu conexión.")

# Botón para forzar actualización
if st.button("🔄 Actualizar con últimos resultados de 2026"):
    st.cache_data.clear()
    st.rerun()
