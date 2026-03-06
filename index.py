import streamlit as st
import pandas as pd
import fastf1
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np

st.set_page_config(page_title="F1 Predictor", layout="wide")
st.title("🏎️ F1 Podium & Position Predictor API")

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
def fetch_api_data(years=[2024, 2025, 2026]):
    all_results = []
    for year in years:
        for round_num in range(1, 25): 
            try:
                session = fastf1.get_session(year, round_num, 'R')
                session.load(telemetry=True, weather=False, messages=True)
                
                banderas_rojas = len(session.track_status[session.track_status['Message'].str.contains('Red', na=False)])
                es_sprint = 1 if session.event['EventFormat'] == 'sprint' else 0
                
                for _, row in session.results.iterrows():
                    piloto = row['Abbreviation']
                    equipo = row['TeamName']
                    
                    try:
                        laps_piloto = session.laps.pick_driver(piloto)
                        paradas_boxes = len(laps_piloto[laps_piloto['PitOutTime'].notnull()])
                        compuestos = laps_piloto['Compound'].dropna().nunique()
                    except:
                        paradas_boxes = 0
                        compuestos = 1
                        
                    try:
                        vuelta_rapida = laps_piloto.pick_fastest()
                        telemetria = vuelta_rapida.get_car_data()
                        vel_maxima = telemetria['Speed'].max()
                    except:
                        vel_maxima = 0
                    
                    all_results.append({
                        'year': year, 
                        'pilot': piloto,
                        'team': equipo,
                        'grid': row['GridPosition'], 
                        'finish': row['Position'],
                        'circuit': session.event['EventName'],
                        'pit_stops': paradas_boxes,
                        'max_speed': vel_maxima,
                        'red_flags': banderas_rojas,
                        'tyre_compounds': compuestos,
                        'is_sprint': es_sprint
                    })
            except Exception:
                continue
    return pd.DataFrame(all_results)

df = fetch_api_data()

if not df.empty:
    df = df.dropna(subset=['finish', 'grid'])
    df['finish'] = pd.to_numeric(df['finish'], errors='coerce')
    df = df.dropna(subset=['finish'])
    df['is_podium'] = df['finish'].apply(lambda x: 1 if x <= 3 else 0)
    
    for col in ['pit_stops', 'max_speed', 'red_flags', 'tyre_compounds', 'is_sprint']:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0)
    
    le_pilot = LabelEncoder()
    le_circuit = LabelEncoder()
    le_team = LabelEncoder()
    
    todos_los_pilotos = pd.concat([df['pilot'], pd.Series(list(PILOTOS_2026.keys()))]).unique()
    le_pilot.fit(todos_los_pilotos)
    
    todos_los_circuitos = pd.concat([df['circuit'], pd.Series(CALENDARIO_2026)]).unique()
    le_circuit.fit(todos_los_circuitos)
    
    equipos_actuales = [info['equipo'] for info in PILOTOS_2026.values()]
    todos_los_equipos = pd.concat([df['team'], pd.Series(equipos_actuales)]).unique()
    le_team.fit(todos_los_equipos)
    
    df['pilot_id'] = le_pilot.transform(df['pilot'].astype(str))
    df['circuit_id'] = le_circuit.transform(df['circuit'].astype(str))
    df['team_id'] = le_team.transform(df['team'].astype(str))
    
    columnas_modelo = ['grid', 'pilot_id', 'team_id', 'circuit_id', 'year', 'pit_stops', 'max_speed', 'red_flags', 'tyre_compounds', 'is_sprint']
    X = df[columnas_modelo]
    y_class = df['is_podium'] 
    y_reg = df['finish']      

    modelos_class = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Lineal / Logistica": LogisticRegression(max_iter=1000)
    }
    
    modelos_reg = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "Lineal / Logistica": LinearRegression()
    }

    for name in modelos_class.keys():
        modelos_class[name].fit(X, y_class)
        modelos_reg[name].fit(X, y_reg)

    st.sidebar.subheader("📅 Última actualización")
    last_year = df['year'].max()
    last_race = df[df['year'] == last_year].iloc[-1]['circuit']
    st.sidebar.success(f"Datos procesados.\n🏁 {last_race} ({last_year})")

    st.sidebar.markdown("---")
    selected_circuit = st.sidebar.selectbox("Selecciona la próxima carrera:", CALENDARIO_2026)
    c_id = le_circuit.transform([selected_circuit])[0]

    st.subheader(f"📊 Comparativa de Predicciones - {selected_circuit}")
    
    col1, col2, col3 = st.columns(3)
    columnas_UI = [col1, col2, col3]
    nombres_modelos = ["Random Forest", "Gradient Boosting", "Lineal / Logistica"]

    for idx, modelo_nombre in enumerate(nombres_modelos):
        with columnas_UI[idx]:
            st.markdown(f"<h4 style='text-align: center;'>{modelo_nombre}</h4>", unsafe_allow_html=True)
            
            predictions = []
            for pilot_abbr, info in PILOTOS_2026.items():
                p_id = le_pilot.transform([pilot_abbr])[0]
                t_id = le_team.transform([info['equipo']])[0]
                
                historial_piloto = df[df['pilot'] == pilot_abbr]
                avg_grid = historial_piloto['grid'].mean() if not historial_piloto.empty else 15.0
                avg_pit = historial_piloto['pit_stops'].mean() if not historial_piloto.empty else 1.0
                avg_speed = historial_piloto['max_speed'].mean() if not historial_piloto.empty else 300.0
                avg_red = historial_piloto['red_flags'].mean() if not historial_piloto.empty else 0.0
                avg_tyres = historial_piloto['tyre_compounds'].mean() if not historial_piloto.empty else 2.0
                avg_sprint = 0 
                
                features_df = pd.DataFrame(
                    [[avg_grid, p_id, t_id, c_id, 2026, avg_pit, avg_speed, avg_red, avg_tyres, avg_sprint]], 
                    columns=columnas_modelo
                )
                
                prob = modelos_class[modelo_nombre].predict_proba(features_df)[0][1]
                pos_pred = modelos_reg[modelo_nombre].predict(features_df)[0]
                
                predictions.append({
                    "Piloto": info['nombre'], 
                    "Pos_Raw": pos_pred,
                    "Prob. Podio (%)": round(prob * 100, 2)
                })

            df_preds = pd.DataFrame(predictions).sort_values(by=["Pos_Raw", "Prob. Podio (%)"], ascending=[True, False])
            df_preds.index = np.arange(1, len(df_preds) + 1)
            df_preds['Puesto'] = df_preds.index
            
            st.dataframe(df_preds[['Puesto', 'Piloto', 'Prob. Podio (%)']], width='stretch', hide_index=True)

            st.markdown("---")
            modelo_actual = modelos_reg[modelo_nombre]
            
            if modelo_nombre in ["Random Forest", "Gradient Boosting"]:
                importances = modelo_actual.feature_importances_
            else:
                importances = np.abs(modelo_actual.coef_)
                suma_importancias = np.sum(importances)
                if suma_importancias > 0:
                    importances = importances / suma_importancias

            fig, ax = plt.subplots(figsize=(4, 3))
            variables = ['Grid', 'Piloto', 'Equipo', 'Circuito', 'Año', 'Pits', 'Vel. Max', 'Banderas Rojas', 'Neumáticos', 'Sprint']
            
            colors = ['#FF1801' if val == max(importances) else '#808080' for val in importances]
            
            ax.barh(variables, importances, color=colors)
            ax.set_xlabel("Impacto")
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.invert_yaxis() 
            
            plt.tight_layout()
            st.pyplot(fig)

    st.markdown("---")
    st.subheader("🗂️ Dataset unificado descargado")
    st.dataframe(df, width='stretch')

    st.markdown("---")
    st.subheader("📥 Exportar Datos")
    csv_data = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descargar Dataset (CSV)",
        data=csv_data,
        file_name="f1_api_data.csv",
        mime="text/csv"
    )
else:
    st.warning("No se pudieron cargar datos de la API. Verifica tu conexión o intenta más tarde.")
