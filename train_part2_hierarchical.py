import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import pickle
import os
import networkx as nx
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

DATA_DIR = "data/processed"
if not os.path.exists(DATA_DIR): DATA_DIR = "uji_data"
MODEL_DIR = "models"
DEVICE = torch.device('cpu')

try:
    from train_part1_hierarchical_v2 import HierarchicalClassifier
    HAS_WIFI = True
except ImportError:
    HAS_WIFI = False

def get_signatures():
    if not HAS_WIFI: return None
    clf_path = f"{MODEL_DIR}/hierarchical_classifier_best.pth"
    if not os.path.exists(clf_path): return None

    try:
        ckpt = torch.load(clf_path, map_location=DEVICE)
        model = HierarchicalClassifier(128, ckpt['num_buildings'], ckpt['num_floors'], ckpt['num_rooms']).to(DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])
        
        df = pd.read_csv(f"{DATA_DIR}/uji_train.csv")
        with open(f"{MODEL_DIR}/scaler.pkl", 'rb') as f: scaler = pickle.load(f)
        X = scaler.transform(df[[f'CSI_DATA{i}' for i in range(1,129)]].values)
        y = df['label'].values
        
        dataset = TensorDataset(torch.FloatTensor(X.copy()), torch.LongTensor(y))
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        
        sigs_sum = {}
        sigs_cnt = {}
        model.eval()
        with torch.no_grad():
            for Xb, yb in loader:
                feats = model.features(Xb.to(DEVICE)).cpu().numpy()
                for i, lbl in enumerate(yb.numpy()):
                    if lbl not in sigs_sum: sigs_sum[lbl] = np.zeros_like(feats[i]); sigs_cnt[lbl] = 0
                    sigs_sum[lbl] += feats[i]; sigs_cnt[lbl] += 1
        return {k: v/sigs_cnt[k] for k, v in sigs_sum.items()}
    except: return None

def load_aligned_env():
    path_loc = f"{DATA_DIR}/location_info_building0.pkl"
    with open(path_loc, 'rb') as f: locs = pickle.load(f)
    nodes = {l['location_id']: l for l in locs if abs(l.get('latitude', 0)) > 1.0}
    
    path_graph = f"{DATA_DIR}/topology_graph_building0.pkl"
    if not os.path.exists(path_graph):
        print("Error: topology_graph_building0.pkl missing. Run rebuild_graph.py first.")
        exit()
        
    with open(path_graph, 'rb') as f: G = pickle.load(f)
    
    valid_ids = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(valid_ids)}
    idx_to_node = {i: n for i, n in enumerate(valid_ids)}
    
    num_nodes = len(valid_ids)
    dist_matrix = np.full((num_nodes, num_nodes), 999)
    
    adj = {n: list(G.neighbors(n)) for n in valid_ids}
    for start_node in tqdm(valid_ids):
        s_i = node_to_idx[start_node]
        dist_matrix[s_i, s_i] = 0
        queue = [(start_node, 0)]
        visited = {start_node}
        
        while queue:
            curr, d = queue.pop(0)
            if d > 20: continue
            
            c_i = node_to_idx[curr]
            dist_matrix[s_i, c_i] = d
            
            for n in adj[curr]:
                if n not in visited:
                    visited.add(n)
                    queue.append((n, d+1))

    sigs = get_signatures()
    node_sigs = {}
    if sigs:
        for nid in valid_ids:
            s = sigs.get(nid, np.zeros(128))
            node_sigs[nid] = s / (np.linalg.norm(s) + 1e-9)
    else:
        for nid in valid_ids: node_sigs[nid] = np.zeros(128)

    return valid_ids, nodes, adj, dist_matrix, node_sigs

# agent
class HybridAgent(nn.Module):
    def __init__(self, wifi_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6 + 3 + wifi_dim*2, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 6), nn.Softmax(dim=-1)
        )
        self.opt = optim.Adam(self.parameters(), lr=0.0001)

    def act(self, state):
        probs = self.net(torch.FloatTensor(state).to(DEVICE))
        probs = torch.clamp(probs, min=1e-6)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

# training
if __name__ == "__main__":
    valid_ids, nodes_info, adj, dist_matrix, node_sigs = load_aligned_env()
    
    agent = HybridAgent(128).to(DEVICE)
    print("\nStarting Aligned Training...")
    
    dirs = [(0,[1,0,0]),(1,[-1,0,0]),(2,[0,1,0]),(3,[0,-1,0]),(4,[0,0,1]),(5,[0,0,-1])]
    
    curriculum_level = 1 
    success_buffer = []
    best_rate = 0.0
    
    pbar = tqdm(range(5000))
    
    for ep in pbar:
        s_idx = np.random.randint(len(valid_ids))
        
        possible = np.where(dist_matrix[s_idx] == curriculum_level)[0]
        if len(possible) == 0:
             possible = np.where((dist_matrix[s_idx] <= curriculum_level) & (dist_matrix[s_idx] > 0))[0]
             if len(possible) == 0: continue
             
        t_idx = np.random.choice(possible)
        
        curr_id = valid_ids[s_idx]
        targ_id = valid_ids[t_idx]
        
        c_info = nodes_info[curr_id]
        t_info = nodes_info[targ_id]
        
        done = False
        steps = 0
        log_probs = []
        rewards = []
        
        dist_to_goal = np.linalg.norm([c_info['latitude']-t_info['latitude'], c_info['longitude']-t_info['longitude']])
        
        while not done and steps < 20:
            steps += 1
            sensors = [-1.0] * 6
            c_info = nodes_info[curr_id]
            neighbors = adj[curr_id]
            
            for n in neighbors:
                n_info = nodes_info[n]
                d = np.array([n_info['latitude']-c_info['latitude'], n_info['longitude']-c_info['longitude'], n_info['floor']-c_info['floor']])
                d_norm = d / (np.linalg.norm(d)+1e-9)
                
                best_d, best_v = -1, 0.5
                for idx, v in dirs:
                    dot = np.dot(d_norm, v)
                    if abs(v[2])>0: 
                         if np.sign(d[2])==np.sign(v[2]) and abs(d[2])>0: dot=2.0
                         else: dot=-1.0
                    if dot > best_v: best_v, best_d = dot, idx
                
                if best_d != -1:
                    nd = np.linalg.norm([n_info['latitude']-t_info['latitude'], n_info['longitude']-t_info['longitude']])
                    sensors[best_d] = 1.0 if nd < dist_to_goal else -0.5

            gps = [(t_info['latitude']-c_info['latitude'])/200.0, (t_info['longitude']-c_info['longitude'])/200.0, (t_info['floor']-c_info['floor'])/5.0]
            
            state = np.concatenate([sensors, gps, node_sigs[curr_id], node_sigs[targ_id]])
            
            action, lp = agent.act(state)
            log_probs.append(lp)
            
            act_vec = dirs[action][1]
            best_n, best_s = None, -99
            
            for n in neighbors:
                n_info = nodes_info[n]
                d = np.array([n_info['latitude']-c_info['latitude'], n_info['longitude']-c_info['longitude'], n_info['floor']-c_info['floor']])
                sim = np.dot(d/(np.linalg.norm(d)+1e-9), act_vec)
                if abs(act_vec[2])>0 and np.sign(d[2])==np.sign(act_vec[2]): sim=2.0
                if sim > best_s: best_s, best_n = sim, n
            
            r = -0.01
            if best_n is not None and best_s > 0.0:
                curr_id = best_n
                c_info = nodes_info[curr_id]
                new_d = np.linalg.norm([c_info['latitude']-t_info['latitude'], c_info['longitude']-t_info['longitude']])
                
                r += np.clip(dist_to_goal - new_d, -1.0, 1.0)
                dist_to_goal = new_d
                
                if curr_id == targ_id:
                    r += 5.0
                    done = True
            else:
                r -= 0.5 # Hit wall
                
            rewards.append(r)
            
        success_buffer.append(1 if curr_id == targ_id else 0)
        
        if len(log_probs) > 0:
            rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
            if rewards.std() > 1e-5: rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            loss = 0
            for lp, r in zip(log_probs, rewards): loss -= lp * r
            agent.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            agent.opt.step()
            
        if (ep+1) % 50 == 0:
            rate = np.mean(success_buffer[-50:])
            if rate > best_rate: best_rate = rate
            pbar.set_description(f"Lvl {curriculum_level} | Succ: {rate:.1%} | Best: {best_rate:.1%}")
            
            if rate > 0.8:
                curriculum_level += 1
                success_buffer = []
                
    print(f"\nFinal Level: {curriculum_level}")
    torch.save(agent.state_dict(), f"{MODEL_DIR}/final_hybrid_agent.pth")