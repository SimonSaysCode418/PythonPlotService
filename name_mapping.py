name_dict = {
    'difference_kwh': 'Abweichung (kWh)',
    20: 'Geschwindigkeit (km/h)',
    271: 'Batterieleistung (kW)',
    4628: 'Leistung Antriebsstrang (kW)',
}


def get_name(key):
    return name_dict.get(key, key)
