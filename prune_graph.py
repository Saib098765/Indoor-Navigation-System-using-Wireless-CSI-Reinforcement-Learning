"""
prune_graph.py
==============
Fixes the "Hairball" graph by removing impossible edges.
1. Removes edges longer than 8 meters (walls/courtyards).
2. Limits max neighbors to 5 (prevents hubs).
3. Ensures the graph stays connected.
"""

import pickle
import networkx as nx
import numpy as np
import os
import matplotlib.pyplot as plt

DATA_DIR = "data/processed"
if not os.path.exists(DATA_DIR): DATA_DIR = "uji_data"

def prune():
    print("\n" + "="*80)
    print("PRUNING GRAPH HAIRBALL")
    print("="*80)
    
    print(f"Loading from {DATA_DIR}...")
    with open(f"{DATA_DIR}/location_info_building0.pkl", 'rb') as f:
        locs = pickle.load(f)
    with open(f"{DATA_DIR}/topology_graph_building0.pkl", 'rb') as f:
        G = pickle.load(f)
        
    print(f"Original: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    loc_map = {l['location_id']: l for l in locs}
    edges_to_remove = []
    distances = []
    
    MAX_DIST_METERS = 8.0  
    
    for u, v in G.edges():
        if u not in loc_map or v not in loc_map:
            edges_to_remove.append((u, v))
            continue
            
        u_info = loc_map[u]
        v_info = loc_map[v]
        
        d_lat = (u_info.get('latitude',0) - v_info.get('latitude',0))
        d_lon = (u_info.get('longitude',0) - v_info.get('longitude',0))
        
        dist = np.sqrt(d_lat**2 + d_lon**2)
        distances.append(dist)
        
        if dist > MAX_DIST_METERS:
            edges_to_remove.append((u, v))
            
        if abs(u_info.get('floor',0) - v_info.get('floor',0)) > 0:
            if dist > 5.0:
                edges_to_remove.append((u, v))

    print(f"Removing {len(edges_to_remove)} long/impossible edges...")
    G.remove_edges_from(edges_to_remove)
    
    # K-NEAREST PRUNING
    pruned_neighbors = 0
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        if len(neighbors) > 6:
            n_dists = []
            u_info = loc_map[node]
            for n in neighbors:
                v_info = loc_map[n]
                d = np.sqrt((u_info.get('latitude',0)-v_info.get('latitude',0))**2 + 
                            (u_info.get('longitude',0)-v_info.get('longitude',0))**2)
                n_dists.append((n, d))
            
            # Keep only closest 4
            n_dists.sort(key=lambda x: x[1])
            to_cut = n_dists[5:] # Keep closest 5
            for n_cut, _ in to_cut:
                if G.has_edge(node, n_cut):
                    G.remove_edge(node, n_cut)
                    pruned_neighbors += 1
                    
    print(f"Pruned {pruned_neighbors} excess neighbor connections.")
    
    if not nx.is_connected(G):
        print("Re-connecting isolated islands...")
        comps = list(nx.connected_components(G))
        main_comp = max(comps, key=len)
        
        for comp in comps:
            if comp == main_comp: continue
            min_dist = 9999
            best_pair = None
            island_sample = list(comp)[:10]
            main_sample = list(main_comp)
            for u in island_sample:
                u_info = loc_map[u]
                for v in locs: 
                    if v['location_id'] in main_comp:
                        v_info = v
                        d = np.sqrt((u_info.get('latitude',0)-v_info.get('latitude',0))**2 + 
                                    (u_info.get('longitude',0)-v_info.get('longitude',0))**2)
                        if d < min_dist:
                            min_dist = d
                            best_pair = (u, v['location_id'])
            
            if best_pair:
                G.add_edge(best_pair[0], best_pair[1])
                print(f"  Bridged island (dist {min_dist:.1f})")

    print(f"Final Graph: {G.number_of_edges()} edges.")
    
    for target_dir in ["data/processed", "uji_data"]:
        if os.path.exists(target_dir):
            with open(f"{target_dir}/topology_graph_building0.pkl", 'wb') as f:
                pickle.dump(G, f)

    try:
        plt.figure()
        plt.hist(distances, bins=50)
        plt.title("Edge Distances (Before Pruning)")
        plt.savefig("edge_dist_hist.png")
    except: pass

if __name__ == "__main__":
    prune()