name_dict = {
    'difference_kwh': 'Abweichung (kWh)',
    'resid': 'Residuen',
    20: 'Geschwindigkeit (km/h)',
    507: 'GPS-Geschwindigkeit (km/h)',
    271: 'Batterieleistung (kW)',
    4628: 'Leistung Antriebsstrang (kW)',
    'train_losses': 'Train Loss',
    'valid_losses': 'Valid Loss',
    'r2_scores': 'R²',
    'delta_forecast_kwh_rel': 'rel. Abweichung',
    'difference_rel': 'rel. Abweichung',
    'target': 'Batterie-Verbrauch (kWh)',
    'prediction': 'Prognose-Verbrauch (kWh)',
    'battery_kwh': 'Batterie-Verbrauch (kWh)',
    'section': 'Sektion (von-nach)',
    'duration_s': 'Fahrdauer (s)',
    'distance_m': 'gemessene Fahrzeugdistanz (m)',
    'weekday': 'Wochentag',
    'hour': 'Stunde des Tages',
    'deviation_section': 'Ø Abweichung von Sektion-Mittelwert (kWh)',
    'outlier_rate': 'Anteil Ausreißer'
}


def get_name(key):
    return name_dict.get(key, str(key))
