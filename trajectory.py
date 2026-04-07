import torch
import numpy as np
import pickle
import os
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = "data/processed"
if not os.path.exists(DATA_DIR): DATA_DIR = "uji_data"
MODEL_DIR = "models"
DEVICE = torch.device('cpu')

def load_system():
    path_loc = f"{DATA_DIR}/location_info_building0.pkl"
    with open(path_loc, 'rb') as f: locs = pickle.load(f)
    nodes = {l['location_id']: l for l in locs if abs(l.get('latitude', 0)) > 1.0}
    
    path_graph = f"{DATA_DIR}/topology_graph_building0.pkl"
    with open(path_graph, 'rb') as f: G = pickle.load(f)

    from train_part1_hierarchical_v2 import HierarchicalClassifier
    ckpt = torch.load(f"{MODEL_DIR}/hierarchical_classifier_best.pth", map_location=DEVICE)
    model = HierarchicalClassifier(128, ckpt['num_buildings'], ckpt['num_floors'], ckpt['num_rooms']).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    df = pd.read_csv(f"{DATA_DIR}/uji_train.csv")
    with open(f"{MODEL_DIR}/scaler.pkl", 'rb') as f: scaler = pickle.load(f)
    X = scaler.transform(df[[f'CSI_DATA{i}' for i in range(1,129)]].values)
    y = df['label'].values
    
    node_wifi_map = {}
    for i in range(len(y)):
        lbl = y[i]
        if lbl not in node_wifi_map: node_wifi_map[lbl] = []
        node_wifi_map[lbl].append(X[i])
        
    return nodes, G, model, node_wifi_map

def plot_trajectory():
    nodes, G, model, node_wifi_map = load_system()
    valid_ids = list(G.nodes())
    path = []
    while len(path) < 20:
        s = np.random.choice(valid_ids)
        t = np.random.choice(valid_ids)
        try:
            path = nx.shortest_path(G, s, t, weight='weight')
        except: continue
        
    gt_coords = []
    pred_coords = []
    
    correct = 0
    
    for node_id in path:
        if node_id not in node_wifi_map: continue
        gt_node = nodes[node_id]
        gt_coords.append([gt_node['longitude'], gt_node['latitude']])
        scans = node_wifi_map[node_id]
        scan = scans[np.random.randint(len(scans))]
        
        with torch.no_grad():
            tensor_x = torch.FloatTensor(np.array([scan])).to(DEVICE)
            output = model(tensor_x)
            if isinstance(output, tuple): _, _, r_logits = output
            else: r_logits = output
            pred_id = torch.argmax(r_logits, dim=1).item()
            
        if pred_id == node_id: correct += 1
            
        if pred_id in nodes:
            p_node = nodes[pred_id]
            pred_coords.append([p_node['longitude'], p_node['latitude']])
        else:
            pred_coords.append(gt_coords[-1]) # Fallback
            
    gt_coords = np.array(gt_coords)
    pred_coords = np.array(pred_coords)
    plt.figure(figsize=(10, 8))
    
    all_longs = [n['longitude'] for n in nodes.values()]
    all_lats = [n['latitude'] for n in nodes.values()]
    plt.scatter(all_longs, all_lats, c='lightgray', s=10, label='Map Nodes')
    plt.plot(gt_coords[:,0], gt_coords[:,1], c='black', linewidth=2, label='True Path')
    plt.scatter(gt_coords[:,0], gt_coords[:,1], c='black', s=30)
    plt.scatter(pred_coords[:,0], pred_coords[:,1], c='red', s=40, marker='x', label='AI Prediction')
    
    for i in range(len(gt_coords)):
        plt.plot([gt_coords[i,0], pred_coords[i,0]], 
                 [gt_coords[i,1], pred_coords[i,1]], 
                 c='red', linestyle='--', linewidth=0.5, alpha=0.5)

    acc = correct / len(path)
    plt.title(f"{acc:.1%} Accuracy on Random Trajectory")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("trajectory_plot.png", dpi=300)

if __name__ == "__main__":
    plot_trajectory()