name_dict = {
    'difference_kwh': 'Abweichung (kWh)',
    'resid': 'Residuen',
    'fittedvalues': 'angepasste Werte',
    'resid_energy': 'Residuen - Energie-Reg',
    # 20: 'Geschwindigkeit (km/h)',
    # 507: 'GPS-Geschwindigkeit (km/h)',
    # 271: 'Batterieleistung (kW)',
    # 4628: 'Leistung Antriebsstrang (kW)',
    'train_losses': 'Train Loss',
    'valid_losses': 'Valid Loss',
    'r2_scores': 'R²',
    'delta_forecast_kwh_rel': 'rel. Abweichung',
    'difference_rel': 'rel. Abweichung',
    'difference_rel_energy': 'rel. Abweichung',
    'target': 'Batterie-Verbrauch (kWh)',
    'prediction': 'Prognose-Verbrauch (kWh)',
    'prediction_energy': 'Prognose-Verbrauch - Energie-Reg (kWh)',
    'battery_kwh': 'Batterie-Verbrauch (kWh)',
    'section': 'Sektion (von-nach)',
    'duration_s': 'Fahrdauer (s)',
    'deviation_section': 'Ø Abweichung von Sektion-Mittelwert (kWh)',
    'outlier_rate': 'Anteil Ausreißer',
    'is_overland': 'Ist Überland',

    'weekday': 'Wochentag',
    'weekday_sin': 'Wochentag (sin)',
    'weekday_cos': 'Wochentag (cos)',
    'hour': 'Stunde des Tages',
    'hour_sin': "Stunde (sin)",
    "hour_cos": "Stunde (cos)",

    'distance_m': 'Fahrzeugdistanz (m)',
    'distance_osm_m': 'OSM-Distanz-OSM (m)',
    'angle_rad_sum': 'Kurven (rad)',
    'ascent_m_sum': 'Aufstieg (m)',
    'descent_m_sum': 'Abstieg (m)',
    'mean_gradient': 'Mittlere Steigung',
    'speed_m_s_max': 'max. Geschwindigkeit (m/s)',
    'speed_m_s_avg': 'Ø Geschwindigkeit (m/s)',
    'speed_m_s_std': 'σ Geschwindigkeit (m/s)',
    'speed_osm_m_s_avg': 'Ø max. OSM-Geschw. (m/s)',
    "temp_outside_avg": 'Außentemperatur (°C)',

    'lat': 'Latitude',
    'lon': 'Longitude',
    'elevation': 'Höhe (m)'
}


def get_name(key):
    return name_dict.get(key, str(key))
