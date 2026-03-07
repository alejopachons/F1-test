import fastf1
import pandas as pd
from data_ingestion import get_current_weekend_fp_data, get_session_laps
from pace_analyzer import analyze_qualifying_pace, analyze_race_pace
from pit_stop_analyzer import analyze_pit_stops
import warnings
import datetime

# Suprimir warnings molestos de pandas/fastf1 en la consola
warnings.filterwarnings('ignore')

def main():
    print("=========================================================")
    print("      PREDICCIONES F1 2026 - (Basado en Prácticas)       ")
    print("=========================================================\n")
    
    año_actual = datetime.datetime.now().year
    if año_actual < 2026:
        # Para propósitos de testing si el sistema corre antes de 2026
        # forzamos a 2026 y advertimos. FastF1 podría fallar si no hay schedule de 2026.
        año_actual = 2026
        
    print(f"Obteniendo datos de la temporada {año_actual}...\n")
    
    # 1. Ingesta de Datos
    fp_data, event = get_current_weekend_fp_data(year=año_actual)
    
    if not fp_data:
        print("No se encontraron datos de sesiones de Práctica Libre para evaluar.")
        print("Asegúrate de que ya haya comenzado la temporada 2026 y haya internet.")
        return

    # Uniremos las vueltas de todas las sesiones de práctica disponibles para más consistencia
    all_laps_list = []
    pit_laps_list = []
    
    for session_name, session in fp_data.items():
        laps = get_session_laps(session)
        if not laps.empty:
            all_laps_list.append(laps)
            pit_laps_list.append(laps)

    if not all_laps_list:
        print("No hay datos de telemetría disponibles en las sesiones descargadas.")
        return

    combined_laps = pd.concat(all_laps_list, ignore_index=True)
    
    # 2. Predicción de Pole (Short Runs en Softs)
    print("\n--- PREDICCIÓN DE POLE POSITION ---")
    quali_df = analyze_qualifying_pace(combined_laps)
    if not quali_df.empty:
        pole_winner = quali_df.iloc[0]
        print(f"🥇 Candidato a la Pole: {pole_winner['Driver']} ({pole_winner['Team']})")
        print("Top 3 a una vuelta (Ideal Lap):")
        for i, row in quali_df.head(3).iterrows():
            print(f"  {i+1}. {row['Driver']} - {row['IdealLap']}")
    else:
        print("No hay suficientes datos de clasificación (Softs) para predecir la pole.")

    # 3. Predicción de Carrera (Podio y 10mo)
    print("\n--- PREDICCIÓN DE CARRERA (Long Runs) ---")
    race_df = analyze_race_pace(combined_laps)
    if not race_df.empty and len(race_df) >= 10:
        print("🏆 Candidatos al Podio:")
        print(f"  1er Lugar: {race_df.iloc[0]['Driver']} ({race_df.iloc[0]['Team']})")
        print(f"  2do Lugar: {race_df.iloc[1]['Driver']} ({race_df.iloc[1]['Team']})")
        print(f"  3er Lugar: {race_df.iloc[2]['Driver']} ({race_df.iloc[2]['Team']})")
        
        print("\n🎯 Candidato al 10mo Lugar (Último lugar en los puntos):")
        # El índice 9 corresponde a la posición 10
        print(f"  10mo Lugar: {race_df.iloc[9]['Driver']} ({race_df.iloc[9]['Team']})")
    else:
        print("No hay suficientes datos de Long Runs (simulaciones de carrera) para predecir podio y 10mo lugar.")

    # 4. Predicción de Pit Stop Rápido
    print("\n--- PREDICCIÓN DE PIT STOP MÁS RÁPIDO ---")
    pit_df = analyze_pit_stops(combined_laps)
    if not pit_df.empty:
        best_pit_team = pit_df.iloc[0]['Team']
        print(f"⏱️  Equipo con paradas más eficientes esperadas: {best_pit_team}")
    else:
        print("No hay suficientes datos de Pit Lane para evaluar tiempos de parada.")

    print("\n=========================================================")
    print("      Fin de las predicciones basadas en Prácticas       ")
    print("=========================================================")

if __name__ == "__main__":
    main()
