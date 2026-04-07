import json
railway_map = {
    "12": {"lat": 12.9781, "lon": 77.5695, "floor": 0, "neighbors": ["20", "88"]},  # Ticket Counter
    "88": {"lat": 12.9783, "lon": 77.5697, "floor": 0, "neighbors": ["12"]},        # Restrooms
    "20": {"lat": 12.9785, "lon": 77.5699, "floor": 0, "neighbors": ["12", "50", "159"]}, # Main Concourse
    "159": {"lat": 12.9780, "lon": 77.5690, "floor": 0, "neighbors": ["20"]},       # Station Entrance
    "50": {"lat": 12.9788, "lon": 77.5702, "floor": 1, "neighbors": ["20"]}         # Platform 1
}
with open('railway_map.json', 'w') as f:
    json.dump(railway_map, f, indent=4)