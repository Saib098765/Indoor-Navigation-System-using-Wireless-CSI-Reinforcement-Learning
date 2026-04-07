# for raw topology graph.
import pickle
import networkx as nx
import os

if os.path.exists("data/processed/location_info.pkl"):
    DATA_DIR = "data/processed"
else:
    DATA_DIR = "uji_data"

def create_single_building_subset():
    print("\n" + "="*80)
    print(f"CREATING DATASET FROM: {DATA_DIR}")
    print("="*80)

    print("Loading raw files...")
    with open(f"{DATA_DIR}/location_info.pkl", 'rb') as f:
        all_locs = pickle.load(f)
    
    with open(f"{DATA_DIR}/topology_graph.pkl", 'rb') as f:
        full_graph = pickle.load(f)

    loc_lookup = {loc['location_id']: loc for loc in all_locs}
    
    updates = 0
    for node in full_graph.nodes():
        if node in loc_lookup:
            info = loc_lookup[node]
            full_graph.nodes[node]['latitude'] = info.get('latitude', 0.0)
            full_graph.nodes[node]['longitude'] = info.get('longitude', 0.0)
            full_graph.nodes[node]['floor'] = info.get('floor', 0)
            updates += 1
            
    print(f"Updated attributes for {updates} nodes.")

    valid_locs = []
    broken_count = 0
    
    for loc in all_locs:
        if loc['building'] == 0:
            lat = loc.get('latitude', 0.0)
            lon = loc.get('longitude', 0.0)
            if abs(lat) > 0.1 and abs(lon) > 0.1:
                valid_locs.append(loc)
            else:
                broken_count += 1
                
    valid_ids = set(loc['location_id'] for loc in valid_locs)
    
    print(f"valid locations in Building 0: {len(valid_locs)}")
    if broken_count > 0:
        print(f"Removed nodes: {broken_count}")
    
    subgraph = full_graph.subgraph(valid_ids).copy()
    
    print(f"Subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")

    targets = ["data/processed", "uji_data"]
    
    for target_dir in targets:
        if os.path.exists(target_dir):
            with open(f"{target_dir}/location_info_building0.pkl", 'wb') as f:
                pickle.dump(valid_locs, f)
            with open(f"{target_dir}/topology_graph_building0.pkl", 'wb') as f:
                pickle.dump(subgraph, f)
            print(f"✓ Saved to {target_dir}")

if __name__ == "__main__":
    create_single_building_subset()