# visualize building0_map
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import os

DATA_DIR = "uji_data"

def visualize():
    try:
        with open(f"{DATA_DIR}/location_info_building0.pkl", 'rb') as f:
            locs = pickle.load(f)
        with open(f"{DATA_DIR}/topology_graph_building0.pkl", 'rb') as f:
            G = pickle.load(f)
    except FileNotFoundError:
        print("Files not found. Run create_building0_subset.py first.")
        return

    print(f"Graph has {len(G.nodes())} nodes and {len(G.edges())} edges.")
    
    node_pos = {}
    node_colors = []
    floors = []
    
    loc_map = {l['location_id']: l for l in locs}

    for node in G.nodes():
        if node in loc_map:
            l = loc_map[node]
            # store as (longitude, latitude) for plotting
            node_pos[node] = (l.get('longitude', 0), l.get('latitude', 0))
            floors.append(l.get('floor', 0))
        else:
            # If node missing in info, put it at 0,0
            node_pos[node] = (0, 0)
            floors.append(-1)

    plt.figure(figsize=(12, 12))
    nx.draw_networkx_edges(G, pos=node_pos, alpha=0.3, edge_color='gray')
    scatter = nx.draw_networkx_nodes(
        G, pos=node_pos, 
        node_size=30, 
        node_color=floors, 
        cmap=plt.cm.get_cmap('jet', 5),
        alpha=0.8
    )
    
    plt.colorbar(scatter, label='Floor Number')
    plt.title("Building 0 Connectivity Map")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True)
    
    output_file = "building0_map.png"
    plt.savefig(output_file, dpi=150)

if __name__ == "__main__":
    visualize()