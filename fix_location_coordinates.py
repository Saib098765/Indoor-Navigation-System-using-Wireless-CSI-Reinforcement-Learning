import pandas as pd
import pickle
import os
import numpy as np

DATA_DIR = "data/processed" 
RAW_TRAIN_PATH = "/Users/saisbala/Desktop/THESIS/uji_raw_download/UJIndoorLoc/trainingData.csv"

def patch_coordinates():
    print("\n" + "="*80)
    print("PATCHING COORDINATES")
    print("="*80)
    
    info_path = os.path.join(DATA_DIR, "location_info.pkl")
    if not os.path.exists(info_path):
        info_path = os.path.join("uji_data", "location_info.pkl")
    
    print(f"Loading metadata from: {info_path}")
    with open(info_path, 'rb') as f:
        location_info = pickle.load(f)
    
    if not os.path.exists(RAW_TRAIN_PATH):
        raise FileNotFoundError(f"Raw file missing at: {RAW_TRAIN_PATH}")
        
    raw_df = pd.read_csv(RAW_TRAIN_PATH)
    
    raw_df['KEY'] = (
        raw_df['BUILDINGID'].astype(int).astype(str) + '_' +
        raw_df['FLOOR'].astype(int).astype(str) + '_' +
        raw_df['SPACEID'].astype(int).astype(str)
    )
    
    coords_map = raw_df.groupby('KEY')[['LATITUDE', 'LONGITUDE']].mean().to_dict('index')
    
    updated = 0
    missing = 0
    
    for loc in location_info:
        b = str(int(loc['building']))
        f = str(int(loc['floor']))
        s = str(int(loc['space']))
        
        key = f"{b}_{f}_{s}"
        
        if key in coords_map:
            loc['latitude'] = coords_map[key]['LATITUDE']
            loc['longitude'] = coords_map[key]['LONGITUDE']
            updated += 1
        else:
            missing += 1
            if 'latitude' not in loc:
                loc['latitude'] = 0.0
                loc['longitude'] = 0.0

    with open(info_path, 'wb') as f:
        pickle.dump(location_info, f)
        
    print(f"\n Updated {updated} locations.")
    print(f"Missing {missing} locations (using defaults).")
    
    sample = location_info[0]
    print(f"\nVerification (Sample Node):")
    print(f"  ID: {sample['location_id']}")
    print(f"  Lat: {sample.get('latitude')}")
    print(f"  Lon: {sample.get('longitude')}")
    
    if sample.get('latitude', 0) == 0:
         print("STILL BROKEN! Sample has 0.0 coordinates.")
    else:
         print("SUCCESS! Coordinates exist.")

if __name__ == "__main__":
    patch_coordinates()