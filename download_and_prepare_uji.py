import pandas as pd
import numpy as np
import os
import pickle
import requests
import zipfile
import io
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import defaultdict
import networkx as nx

OUTPUT_DIR = "uji_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_uji_dataset():
    print("\n" + "="*80)
    print("="*80)
    url = "https://archive.ics.uci.edu/static/public/310/ujiindoorloc.zip"
    save_dir = "uji_raw_download"
    os.makedirs(save_dir, exist_ok=True)
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(save_dir)
            
        train_path = f"{save_dir}/trainingData.csv"
        val_path = f"{save_dir}/validationData.csv"
        
        if not os.path.exists(train_path):
            for root, dirs, files in os.walk(save_dir):
                if 'trainingData.csv' in files:
                    train_path = os.path.join(root, 'trainingData.csv')
                    val_path = os.path.join(root, 'validationData.csv')
                    break
        
        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        df = pd.concat([df_train, df_val], ignore_index=True)
        return df

    except Exception as e:
        print(f"\nDownload failed {e}")
        raise e

def analyze_uji_structure(df):
    print("\n" + "="*80)
    print("="*80)
    print(f"\nTotal samples: {len(df)}")
    print(f"Features: {len([c for c in df.columns if c.startswith('WAP')])}")
    
    buildings = sorted(df['BUILDINGID'].unique())
    print(f"\nBuildings: {buildings}")
    print("\nFloors per building:")
    for building in buildings:
        floors = sorted(df[df['BUILDINGID'] == building]['FLOOR'].unique())
        print(f"  Building {building}: Floors {floors}")
    
    if 'SPACEID' in df.columns:
        print(f"\nSpace IDs available: {df['SPACEID'].nunique()} unique")
        use_spaceid = True
    else:
        print("\nNo SPACEID column found, use grid-based locations")
        use_spaceid = False
    
    print("\nSamples per building/floor:")
    for building in buildings:
        for floor in sorted(df[df['BUILDINGID'] == building]['FLOOR'].unique()):
            count = len(df[(df['BUILDINGID'] == building) & (df['FLOOR'] == floor)])
            print(f"  Building {building}, Floor {floor}: {count} samples")
    
    return {
        'buildings': buildings,
        'use_spaceid': use_spaceid,
        'total_samples': len(df)
    }

def create_location_labels(df, use_spaceid=True):
    print("\n" + "="*80)
    print("="*80)
    
    location_map = {}
    location_info = []
    
    if use_spaceid and 'SPACEID' in df.columns:
        print("\nUsing SPACEID for location labels...")
        
        df['LOCATION_ID'] = (
            df['BUILDINGID'].astype(str) + '_' +
            df['FLOOR'].astype(str) + '_' +
            df['SPACEID'].astype(str)
        )
        
        unique_locations = sorted(df['LOCATION_ID'].unique())
        location_map = {loc: idx for idx, loc in enumerate(unique_locations)}
        
        for loc_str, loc_id in location_map.items():
            parts = loc_str.split('_')
            location_info.append({
                'location_id': loc_id,
                'building': int(parts[0]),
                'floor': int(parts[1]),
                'space': parts[2],
                'type': 'spaceid'
            })
        
        df['LABEL'] = df['LOCATION_ID'].map(location_map)
        
    else:
        print("\nUsing grid-based location labels")
        
        label_counter = 0
        
        for building in sorted(df['BUILDINGID'].unique()):
            for floor in sorted(df[df['BUILDINGID'] == building]['FLOOR'].unique()):
                mask = (df['BUILDINGID'] == building) & (df['FLOOR'] == floor)
                subset = df[mask]

                lat_min, lat_max = subset['LATITUDE'].min(), subset['LATITUDE'].max()
                lon_min, lon_max = subset['LONGITUDE'].min(), subset['LONGITUDE'].max()
                grid_size = 3.0
                lat_bins = np.arange(lat_min, lat_max + grid_size, grid_size)
                lon_bins = np.arange(lon_min, lon_max + grid_size, grid_size)
                lat_grid = np.digitize(subset['LATITUDE'], lat_bins)
                lon_grid = np.digitize(subset['LONGITUDE'], lon_bins)
                grid_cells = lat_grid * 1000 + lon_grid
                unique_cells = sorted(pd.unique(grid_cells))
                
                for cell in unique_cells:
                    cell_mask = (grid_cells == cell)
                    df.loc[mask & cell_mask, 'LABEL'] = label_counter

                    cell_data = subset[cell_mask.values].iloc[0]
                    location_info.append({
                        'location_id': label_counter,
                        'building': building,
                        'floor': floor,
                        'latitude': cell_data['LATITUDE'],
                        'longitude': cell_data['LONGITUDE'],
                        'type': 'grid'
                    })
                    
                    label_counter += 1
    
    print("\nLocation distribution:")
    for building in sorted(df['BUILDINGID'].unique()):
        for floor in sorted(df[df['BUILDINGID'] == building]['FLOOR'].unique()):
            count = len(df[(df['BUILDINGID'] == building) & (df['FLOOR'] == floor)]['LABEL'].unique())
            print(f"  Building {building}, Floor {floor}: {count} locations")
    
    return df, location_info

def convert_to_project_format(df):

    print("\n" + "="*80)
    print("="*80)
    
    wap_columns = [col for col in df.columns if col.startswith('WAP')]
    print(f"\nFound {len(wap_columns)} WAP (RSSI) features")
    
    converted_data = []
    
    for idx, row in df.iterrows():
        # Extract RSSI values (WAP001-WAP520)
        rssi_values = row[wap_columns].values
        # Replace 100 (not detected) with -104 (very weak signal)
        rssi_values = np.where(rssi_values == 100, -104, rssi_values)
        # Normalize RSSI to positive values for CSI format, RSSI ranges from -104 to 0, convert to 0-104
        rssi_normalized = rssi_values + 104
        # Pad or truncate to 128 "CSI" values
        if len(rssi_normalized) < 128:
            # Interpolate to 128
            csi_values = np.interp(
                np.linspace(0, len(rssi_normalized)-1, 128),
                np.arange(len(rssi_normalized)),
                rssi_normalized
            )
        else:
            indices = np.linspace(0, len(rssi_normalized)-1, 128, dtype=int)
            csi_values = rssi_normalized[indices]
        
        record = {
            'type': 'CSI_DATA',
            'role': 'PASSIVE',
            'mac': '00:00:00:00:00:00',
            'rssi': np.mean(rssi_values[rssi_values != -104]),  # Average of detected signals
            'rate': 11.0,
            'sig_mode': 0.0,
            'mcs': 0.0,
            'bandwidth': 20.0,
            'smoothing': 0.0,
            'not_sounding': 0.0,
            'aggregation': 0.0,
            'stbc': 0.0,
            'fec_coding': 0.0,
            'sgi': 0.0,
            'noise_floor': -90.0,
            'ampdu_cnt': 0.0,
            'channel': 6.0,
            'secondary_channel': 0.0,
            'local_timestamp': idx,
            'ant': 0.0,
            'sig_len': 128.0,
            'rx_state': 0.0,
            'real_time_set': 0.0,
            'real_timestamp': idx,
            'len': 128.0,
        }
        
        for i in range(128):
            record[f'CSI_DATA{i+1}'] = csi_values[i]
        
        record['timestamp'] = idx
        record['label'] = row['LABEL']
        record['building'] = row['BUILDINGID']
        record['floor'] = row['FLOOR']
        record['latitude'] = row['LATITUDE']
        record['longitude'] = row['LONGITUDE']
        
        converted_data.append(record)
        
        if (idx + 1) % 1000 == 0:
            print(f"  Converted {idx + 1}/{len(df)} samples...")
    
    converted_df = pd.DataFrame(converted_data)
    
    print(f"\n✓ Converted {len(converted_df)} samples")
    return converted_df

def create_navigation_topology(location_info):
    print("\n" + "="*80)
    print("="*80)
    
    G = nx.Graph()
    
    for loc in location_info:
        G.add_node(loc['location_id'], **loc)
   
    by_building_floor = defaultdict(list)
    for loc in location_info:
        key = (loc['building'], loc['floor'])
        by_building_floor[key].append(loc)

    for (building, floor), locations in by_building_floor.items():
        print(f"  Building {building}, Floor {floor}: {len(locations)} locations")
        
        if len(locations) <= 1:
            continue

        for i, loc1 in enumerate(locations):
            for loc2 in locations[i+1:]:
                if 'latitude' in loc1 and 'longitude' in loc1:
                    dist = np.sqrt(
                        (loc1['latitude'] - loc2['latitude'])**2 +
                        (loc1['longitude'] - loc2['longitude'])**2
                    )
                    
                    if dist < 10.0:
                        G.add_edge(loc1['location_id'], loc2['location_id'], 
                                  weight=dist, type='same_floor')
                else:
                    G.add_edge(loc1['location_id'], loc2['location_id'],
                              weight=1.0, type='same_floor')

    for building in set(loc['building'] for loc in location_info):
        floors_in_building = sorted(set(
            loc['floor'] for loc in location_info if loc['building'] == building
        ))

        for i in range(len(floors_in_building) - 1):
            floor1 = floors_in_building[i]
            floor2 = floors_in_building[i + 1]
            
            locs_f1 = [loc for loc in location_info 
                      if loc['building'] == building and loc['floor'] == floor1]
            locs_f2 = [loc for loc in location_info 
                      if loc['building'] == building and loc['floor'] == floor2]
    
            if locs_f1 and locs_f2:
                G.add_edge(locs_f1[0]['location_id'], locs_f2[0]['location_id'],
                          weight=5.0, type='stairs')
    
    print(f"\nCreated graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    if nx.is_connected(G):
        print("Graph is fully connected (all locations reachable)")
    else:
        components = list(nx.connected_components(G))
        print(f"Graph has {len(components)} disconnected components")
        print(f" Largest component: {len(max(components, key=len))} nodes")
    
    return G

def save_prepared_data(train_df, test_df, location_info, topology_graph):
    print("\n" + "="*80)
    print("="*80)
    
    train_path = f"{OUTPUT_DIR}/uji_train.csv"
    test_path = f"{OUTPUT_DIR}/uji_test.csv"
    
    csv_columns = [c for c in train_df.columns if c not in ['building', 'floor', 'latitude', 'longitude']]
    
    train_df[csv_columns].to_csv(train_path, index=False)
    test_df[csv_columns].to_csv(test_path, index=False)
    
    print(f"Saved training data: {train_path} ({len(train_df)} samples)")
    print(f"Saved test data: {test_path} ({len(test_df)} samples)")
    
    location_info_path = f"{OUTPUT_DIR}/location_info.pkl"
    with open(location_info_path, 'wb') as f:
        pickle.dump(location_info, f)
    print(f"Saved location info: {location_info_path}")
    
    topology_path = f"{OUTPUT_DIR}/topology_graph.pkl"
    with open(topology_path, 'wb') as f:
        pickle.dump(topology_graph, f)
    print(f"Saved topology graph: {topology_path}")
    
    summary = {
        'num_locations': len(location_info),
        'num_buildings': len(set(loc['building'] for loc in location_info)),
        'num_floors': len(set((loc['building'], loc['floor']) for loc in location_info)),
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'topology_nodes': topology_graph.number_of_nodes(),
        'topology_edges': topology_graph.number_of_edges()
    }
    
    summary_path = f"{OUTPUT_DIR}/dataset_summary.pkl"
    with open(summary_path, 'wb') as f:
        pickle.dump(summary, f)
    
    print(f"\n{'='*80}")
    print(f"{'='*80}")
    print(f"\nSummary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    return summary

def main():
    df = download_uji_dataset()
    analysis = analyze_uji_structure(df)
    df, location_info = create_location_labels(df, analysis['use_spaceid'])
    converted_df = convert_to_project_format(df)
    topology_graph = create_navigation_topology(location_info)
    
    print("\n" + "="*80)
    print("="*80)
    
    try:
        train_df, test_df = train_test_split(
            converted_df, 
            test_size=0.2, 
            random_state=42, 
            stratify=converted_df['label']
        )
    except ValueError:
        # small sample size
        train_df, test_df = train_test_split(
            converted_df, 
            test_size=0.2, 
            random_state=42
        )

    train_rooms = set(train_df['label'].unique())
    test_rooms = set(test_df['label'].unique())
    missing_in_train = test_rooms - train_rooms
    
    if missing_in_train:
        print(f"Warning: {len(missing_in_train)} rooms are in Test but NOT in Train.")
    else:
        print("Validation: All Test rooms exist in Training set.")

    summary = save_prepared_data(train_df, test_df, location_info, topology_graph)
    
    print(f"\n{'='*80}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()