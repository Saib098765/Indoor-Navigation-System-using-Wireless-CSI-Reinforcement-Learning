import pickle
import networkx as nx
import numpy as np
import os
from itertools import combinations

DATA_DIR = "data/processed"
if not os.path.exists(DATA_DIR): DATA_DIR = "uji_data"

def rebuild():
    print("\n" + "="*80)
    print("="*80)
    
    # 1. Load Nodes Only (Discard Edges)
    path = f"{DATA_DIR}/location_info_building0.pkl"
    
    with open(path, 'rb') as f:
        locs = pickle.load(f)
        
    G = nx.Graph()
    loc_map = {}
    
    for l in locs:
        if abs(l.get('latitude', 0)) > 1.0:
            nid = l['location_id']
            G.add_node(nid, **l)
            loc_map[nid] = l
                
    floors = set(l['floor'] for l in locs)
    
    for floor in floors:
        floor_nodes = [n for n in G.nodes() if loc_map[n]['floor'] == floor]
        print(f"  Floor {floor}: Processing {len(floor_nodes)} nodes...")
        
        for i, u in enumerate(floor_nodes):
            u_pos = np.array([loc_map[u]['latitude'], loc_map[u]['longitude']])
            distances = []
            
            for v in floor_nodes:
                if u == v: continue
                v_pos = np.array([loc_map[v]['latitude'], loc_map[v]['longitude']])
                dist = np.linalg.norm(u_pos - v_pos)
                distances.append((v, dist))
            
            distances.sort(key=lambda x: x[1])
            added = 0
            for v, dist in distances:
                if dist < 25.0: 
                    G.add_edge(u, v, weight=dist)
                    added += 1
                if added >= 3: break # Limit degree to keep graph clean
    
    for u in G.nodes():
        for v in G.nodes():
            if u >= v: continue
            
            u_info = loc_map[u]
            v_info = loc_map[v]
            
            f_diff = abs(u_info['floor'] - v_info['floor'])
            if f_diff == 1:
                d_lat = u_info['latitude'] - v_info['latitude']
                d_lon = u_info['longitude'] - v_info['longitude']
                dist_2d = np.sqrt(d_lat**2 + d_lon**2)
                
                if dist_2d < 5.0: 
                    G.add_edge(u, v, weight=dist_2d + 10.0) 
                    
    print("  Ensuring full connectivity...")
    if not nx.is_connected(G):
        comps = list(nx.connected_components(G))
        print(f"  Found {len(comps)} disconnected components. Bridging...")
        
        while len(comps) > 1:
            c1 = list(comps[0])
            best_pair = None
            min_dist = 99999.0
            
            for u in c1:
                u_pos = np.array([loc_map[u]['latitude'], loc_map[u]['longitude']])
                
                for other_comp in comps[1:]:
                    for v in other_comp:
                        if loc_map[u]['floor'] != loc_map[v]['floor']: continue
                        
                        v_pos = np.array([loc_map[v]['latitude'], loc_map[v]['longitude']])
                        dist = np.linalg.norm(u_pos - v_pos)
                        
                        if dist < min_dist:
                            min_dist = dist
                            best_pair = (u, v)
                            
            if best_pair:
                G.add_edge(best_pair[0], best_pair[1], weight=min_dist)
                
            comps = list(nx.connected_components(G))

    print(f"Final Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    
    for target_dir in ["data/processed", "uji_data"]:
        if os.path.exists(target_dir):
            path = f"{target_dir}/topology_graph_building0.pkl"
            with open(path, 'wb') as f:
                pickle.dump(G, f)
                
if __name__ == "__main__":
    rebuild()