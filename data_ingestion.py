import fastf1
import pandas as pd
import datetime

# Habilitar caché para no descargar la misma data múltiples veces
fastf1.Cache.enable_cache('cache')

def get_current_weekend_fp_data(year=2026, round_num=None):
    """
    Obtiene los datos de las Prácticas Libres (FP1, FP2, FP3) para el evento actual o especificado.
    Solo admite datos de la temporada 2026 en adelante, descartando los históricos debido
    a las nuevas normativas.
    """
    if year < 2026:
        raise ValueError("Se solicitaron datos de un año anterior a 2026. Las normativas han cambiado, por lo que se ignora el histórico.")

    # Si no se especifica el round, intentar obtener el evento actual (el último que ocurrió)
    # Por defecto, probemos obtener el calendario de 2026 para encontrar el evento.
    try:
        if round_num is None:
            # fastf1 no tiene una función para 'round actual' tan trivial, 
            # buscaremos el evento más reciente basado en la fecha
            schedule = fastf1.get_event_schedule(year)
            now = datetime.datetime.now()
            # Buscar el evento que acaba de ocurrir o está ocurriendo
            past_events = schedule[schedule['EventDate'] < now]
            if not past_events.empty:
                round_num = past_events.iloc[-1]['RoundNumber']
            else:
                # Si no hay eventos pasados (ej. pretemporada), podríamos forzar el round 1 o Testing
                round_num = 1
                print("No hay eventos pasados encontrados para este año. Usando Round 1 por defecto.")
        
        event = fastf1.get_event(year, round_num)
        print(f"Descargando datos para: {event['EventName']} (Round {round_num})")
        
        fp_data = {}
        for session_name in ['FP1', 'FP2', 'FP3']:
            try:
                session = fastf1.get_session(year, round_num, session_name)
                session.load()
                fp_data[session_name] = session
                print(f"Datos de {session_name} cargados exitosamente.")
            except Exception as e:
                print(f"No se pudo cargar {session_name}: {e}")
        
        return fp_data, event

    except Exception as e:
        print(f"Error al obtener los datos del fin de semana: {e}")
        return None, None

def get_session_laps(session):
    """Extrae las vueltas y la telemetría básica de una sesión cargada."""
    if session is None:
        return pd.DataFrame()
    return session.laps
