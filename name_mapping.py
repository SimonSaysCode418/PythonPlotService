name_dict = {
    'difference_kwh': 'Abweichung (kWh)',
    'resid': 'Residuen',
    20: 'Geschwindigkeit (km/h)',
    507: 'GPS-Geschwindigkeit (km/h)',
    271: 'Batterieleistung (kW)',
    4628: 'Leistung Antriebsstrang (kW)',
}


def get_name(key):
    return name_dict.get(key, str(key))
