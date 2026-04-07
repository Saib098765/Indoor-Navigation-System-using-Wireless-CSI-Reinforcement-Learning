import pickle
import networkx as nx
import numpy as np
from collections import defaultdict

DATA_DIR = "uji_data"

def fix_graph_topology():
    print("\n" + "="*80)
    print("="*80)
    
    with open(f"{DATA_DIR}/topology_graph.pkl", 'rb') as f:
        G = pickle.load(f)
    
    with open(f"{DATA_DIR}/location_info.pkl", 'rb') as f:
        location_info = pickle.load(f)
    
    print(f"\nOriginal graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    loc_dict = {loc['location_id']: loc for loc in location_info}
    
    by_building_floor = defaultdict(list)
    for loc_id, loc in loc_dict.items():
        key = (loc['building'], loc['floor'])
        by_building_floor[key].append(loc_id)
    
    for key, locs in sorted(by_building_floor.items()):
        print(f"  Building {key[0]}, Floor {key[1]}: {len(locs)} locations")
    
    edges_added = 0
    
    for (building, floor), nodes in by_building_floor.items():
        if len(nodes) <= 1:
            continue
        
        k = min(4, len(nodes) - 1)  # Connect to 4 nearest neighbors
        
        for node in nodes:
            loc = loc_dict[node]
            
            # same floor neighbours
            if 'latitude' in loc and 'longitude' in loc:
                distances = []
                for other_node in nodes:
                    if other_node == node:
                        continue
                    other_loc = loc_dict[other_node]
                    
                    if 'latitude' in other_loc and 'longitude' in other_loc:
                        dist = np.sqrt(
                            (loc['latitude'] - other_loc['latitude'])**2 +
                            (loc['longitude'] - other_loc['longitude'])**2
                        )
                        distances.append((other_node, dist))
                
                # Connect to k nearest
                distances.sort(key=lambda x: x[1])
                for other_node, dist in distances[:k]:
                    if not G.has_edge(node, other_node):
                        G.add_edge(node, other_node, weight=dist, type='same_floor')
                        edges_added += 1
            else:
                # No coordinates, connect to random neighbors
                others = [n for n in nodes if n != node]
                for other in np.random.choice(others, min(k, len(others)), replace=False):
                    if not G.has_edge(node, other):
                        G.add_edge(node, other, weight=1.0, type='same_floor')
                        edges_added += 1
    
    print(f"  Added {edges_added} intra-floor edges")
    
    print(f"\nStep 2: Adding floor change")
    stairs_added = 0
    
    for building in set(loc['building'] for loc in location_info):
        floors_in_building = sorted(set(
            loc['floor'] for loc in location_info if loc['building'] == building
        ))
        
        for i in range(len(floors_in_building) - 1):
            floor1 = floors_in_building[i]
            floor2 = floors_in_building[i + 1]
            
            nodes_f1 = by_building_floor[(building, floor1)]
            nodes_f2 = by_building_floor[(building, floor2)]
            
            if nodes_f1 and nodes_f2:
                num_connections = min(3, len(nodes_f1), len(nodes_f2))
                
                for _ in range(num_connections):
                    n1 = np.random.choice(nodes_f1)
                    n2 = np.random.choice(nodes_f2)
                    
                    if not G.has_edge(n1, n2):
                        G.add_edge(n1, n2, weight=5.0, type='stairs')
                        stairs_added += 1
    
    print(f" {stairs_added} vertical connections")
    
    # Step 3: Connect buildings
    building_edges = 0
    
    buildings = sorted(set(loc['building'] for loc in location_info))
    
    for i in range(len(buildings)):
        for j in range(i+1, len(buildings)):
            b1, b2 = buildings[i], buildings[j]
            
            nodes_b1_f0 = by_building_floor.get((b1, 0), [])
            nodes_b2_f0 = by_building_floor.get((b2, 0), [])
            
            if nodes_b1_f0 and nodes_b2_f0:
                for _ in range(2):
                    n1 = np.random.choice(nodes_b1_f0)
                    n2 = np.random.choice(nodes_b2_f0)
                    
                    if not G.has_edge(n1, n2):
                        G.add_edge(n1, n2, weight=20.0, type='building_transition')
                        building_edges += 1
    
    print(f"  Added {building_edges} cross-building connections")
        
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        print(f"  Found {len(components)} components")
        
        largest = max(components, key=len)
        
        for component in components:
            if component == largest:
                continue
            
            node_small = np.random.choice(list(component))
            node_large = np.random.choice(list(largest))
            
            G.add_edge(node_small, node_large, weight=10.0, type='bridge')
            print(f"    Connected component (size {len(component)}) to main graph")
    
    # Final statistics
    print(f"\n" + "="*80)
    print("FIXED GRAPH:")
    print("="*80)
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    print(f"  Connected: {nx.is_connected(G)}")
    
    if nx.is_connected(G):
        print(f"  Diameter: {nx.diameter(G)}")
        backup_path = f"{DATA_DIR}/topology_graph_backup.pkl"
    with open(backup_path, 'wb') as f:
        pickle.dump(pickle.load(open(f"{DATA_DIR}/topology_graph.pkl", 'rb')), f)
    print(f"\nBackup saved: {backup_path}")
    
    with open(f"{DATA_DIR}/topology_graph.pkl", 'wb') as f:
        pickle.dump(G, f)
    print(f"Fixed graph saved: {DATA_DIR}/topology_graph.pkl")
    
    print("\nReady to run `python train_part2_hierarchical.py`")

if __name__ == "__main__":
    fix_graph_topology()
