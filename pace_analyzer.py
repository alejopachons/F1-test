import pandas as pd
import numpy as np

def _is_short_run(stint_laps):
    """Consideramos un stint corto (simulación de clasificación) si tiene menos de 5 vueltas cronometradas."""
    return len(stint_laps) <= 5

def analyze_qualifying_pace(session_laps):
    """
    Analiza el ritmo de clasificación (Short Runs).
    Retorna un DataFrame con los pilotos ordenados por su mejor vuelta teórica
    y mejor vuelta real con neumáticos blandos (Soft).
    """
    if session_laps is None or session_laps.empty:
        return pd.DataFrame()
        
    # Filtrar vueltas válidas (sin pit in/out) y con compuesto blando
    valid_laps = session_laps.pick_quicklaps()
    soft_laps = valid_laps[valid_laps['Compound'] == 'SOFT']
    
    if soft_laps.empty:
        print("No se encontraron vueltas con neumáticos blandos válidos. Usando todos los compuestos.")
        soft_laps = valid_laps
        
    # Agrupar por piloto
    drivers = soft_laps['Driver'].unique()
    quali_data = []
    
    for driver in drivers:
        driver_laps = soft_laps.pick_driver(driver)
        if driver_laps.empty:
            continue
            
        # Mejor vuelta real
        best_lap = driver_laps.pick_fastest()
        if pd.isnull(best_lap['LapTime']):
            continue
            
        # Sector times
        best_s1 = driver_laps['Sector1Time'].min()
        best_s2 = driver_laps['Sector2Time'].min()
        best_s3 = driver_laps['Sector3Time'].min()
        
        ideal_lap_time = best_s1 + best_s2 + best_s3
        
        quali_data.append({
            'Driver': driver,
            'Team': best_lap['Team'],
            'BestLap': best_lap['LapTime'],
            'BestLap_Seconds': best_lap['LapTime'].total_seconds(),
            'IdealLap': ideal_lap_time,
            'IdealLap_Seconds': ideal_lap_time.total_seconds() if pd.notnull(ideal_lap_time) else np.nan
        })
        
    df = pd.DataFrame(quali_data)
    if not df.empty:
        df = df.sort_values(by='IdealLap_Seconds')
        df = df.reset_index(drop=True)
        
    return df

def analyze_race_pace(session_laps):
    """
    Analiza el ritmo de carrera (Long Runs).
    Filtra stints largos y calcula el promedio por vuelta excluyendo tráfico/errores.
    Retorna un DataFrame ordenado de mejor a peor ritmo global.
    """
    if session_laps is None or session_laps.empty:
        return pd.DataFrame()
        
    # Agrupar por piloto y por número de stint
    drivers = session_laps['Driver'].unique()
    race_data = []
    
    for driver in drivers:
        driver_laps = session_laps.pick_driver(driver)
        # Filtramos vueltas excesivamente lentas (ej. in/out laps o VSC) usando track status
        track_laps = driver_laps[(driver_laps['TrackStatus'] == '1') & (driver_laps['PitOutTime'].isnull()) & (driver_laps['PitInTime'].isnull())]
        
        stints = track_laps['Stint'].unique()
        long_run_times = []
        
        for stint in stints:
            stint_laps = track_laps[track_laps['Stint'] == stint]
            # Excluimos la primera vuelta lanzada si es muy atípica y consideramos stint "largo" si > 4 vueltas útiles
            if len(stint_laps) > 4:
                # Quitamos el 25% de las vueltas más lentas (tráfico/errores)
                q75 = stint_laps['LapTime'].quantile(0.75)
                clean_laps = stint_laps[stint_laps['LapTime'] < q75]
                if not clean_laps.empty:
                    avg_time = clean_laps['LapTime'].mean()
                    long_run_times.append(avg_time.total_seconds())
                    
        # Promediar todos los stints largos de este piloto
        if long_run_times:
            avg_race_pace = np.mean(long_run_times)
            race_data.append({
                'Driver': driver,
                'Team': track_laps.iloc[0]['Team'] if not track_laps.empty else 'Unknown',
                'AvgRacePace_Seconds': avg_race_pace
            })
            
    df = pd.DataFrame(race_data)
    if not df.empty:
        df = df.sort_values(by='AvgRacePace_Seconds')
        df = df.reset_index(drop=True)
        
    return df
