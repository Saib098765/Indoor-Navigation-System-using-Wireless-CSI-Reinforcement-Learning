"""
plot_trajectory.py
==================
Visualizes a single navigation episode on the synthetic grid campus,
comparing the OQA-PPO agent's path against the BFS shortest path.

Key fix vs previous version:
  - BFS now uses node_to_idx dict (O(1) per node lookup vs O(N) before)
  - Model checkpoint name updated to match OQA-PPO training output
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import os
from train_part2_hierarchical import HybridPPOAgent, DEVICE

MODEL_PATH    = "models/final_hybrid_fappo_agent.pth"
DATA_DIR      = "data/processed"
NUM_LANDMARKS = 8     # Random points of interest scattered on the floor plan
SEED          = 90  # Set to an int (e.g. 42) for reproducible episodes


# ---------------------------------------------------------------------------
# Environment loader (O(N) BFS)
# ---------------------------------------------------------------------------

def load_synthetic_env():
    print("[INFO] Loading Grid Campus...")

    with open(f"{DATA_DIR}/location_info_synthetic.pkl", 'rb') as f:
        locs = pickle.load(f)
    nodes = {l['location_id']: l for l in locs}

    with open(f"{DATA_DIR}/topology_graph_synthetic.pkl", 'rb') as f:
        G = pickle.load(f)

    valid_ids   = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(valid_ids)}     # O(1) lookup
    adj         = {n: list(G.neighbors(n)) for n in valid_ids}

    num_nodes   = len(valid_ids)
    dist_matrix = np.full((num_nodes, num_nodes), 999, dtype=np.int32)

    for start_node in valid_ids:
        s_i     = node_to_idx[start_node]
        dist_matrix[s_i, s_i] = 0
        queue   = [(start_node, 0)]
        visited = {start_node}

        while queue:
            curr, d = queue.pop(0)
            if d > 48:
                continue
            dist_matrix[s_i, node_to_idx[curr]] = d
            for n in adj[curr]:
                if n not in visited:
                    visited.add(n)
                    queue.append((n, d + 1))

    node_sigs = {nid: np.zeros(128) for nid in valid_ids}
    return valid_ids, nodes, adj, dist_matrix, node_sigs, node_to_idx


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

if SEED is not None:
    np.random.seed(SEED)

valid_ids, nodes_info, adj, dist_matrix, node_sigs, node_to_idx = load_synthetic_env()

agent = HybridPPOAgent(128).to(DEVICE)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. "
        "Run train_part2_hierarchical.py first."
    )
agent.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
agent.eval()

dirs = [
    (0, [1, 0, 0]), (1, [-1, 0, 0]),
    (2, [0, 1, 0]), (3, [0, -1, 0]),
    (4, [0, 0, 1]), (5, [0, 0, -1]),
]

# --- Pick start and goal (15-25 hops apart for a visually interesting path) ---
start_idx = np.random.randint(len(valid_ids))
start_id  = valid_ids[start_idx]

possible_goals = np.where(
    (dist_matrix[start_idx] >= 15) & (dist_matrix[start_idx] <= 25)
)[0]
targ_id = (
    valid_ids[np.random.choice(possible_goals)]
    if len(possible_goals) > 0
    else valid_ids[np.random.randint(len(valid_ids))]
)

# --- Scatter random landmarks ---
landmarks = []
while len(landmarks) < NUM_LANDMARKS:
    lm = valid_ids[np.random.randint(len(valid_ids))]
    if lm not in (start_id, targ_id) and lm not in landmarks:
        landmarks.append(lm)

true_hop_dist = dist_matrix[node_to_idx[start_id], node_to_idx[targ_id]]
print(f"\n[INFO] Episode: Start({start_id}) -> Goal({targ_id})")
print(f"[INFO] BFS shortest path: {true_hop_dist} hops")


# ---------------------------------------------------------------------------
# Greedy inference
# ---------------------------------------------------------------------------

agent_path = [start_id]
curr_id    = start_id
visited    = [start_id]
done       = False
total_steps = 0

G_temp         = nx.from_dict_of_lists(adj)
shortest_path  = nx.shortest_path(G_temp, start_id, targ_id)

with torch.no_grad():
    while not done and total_steps < 60:
        total_steps += 1
        c_info = nodes_info[curr_id]
        t_info = nodes_info[targ_id]
        neighbors   = adj[curr_id]
        dist_now = np.linalg.norm([
            c_info['latitude']  - t_info['latitude'],
            c_info['longitude'] - t_info['longitude'],
        ])

        # Action mask
        valid_mask = [0.0] * 6
        for a_idx, (_, v_vec) in enumerate(dirs):
            for n in neighbors:
                n_info = nodes_info[n]
                d = np.array([
                    n_info['latitude']  - c_info['latitude'],
                    n_info['longitude'] - c_info['longitude'],
                    n_info['floor']     - c_info['floor'],
                ])
                sim = np.dot(d / (np.linalg.norm(d) + 1e-9), v_vec)
                if abs(v_vec[2]) > 0 and np.sign(d[2]) == np.sign(v_vec[2]):
                    sim = 2.0
                if sim > 0.0:
                    valid_mask[a_idx] = 1.0
                    break
        if sum(valid_mask) == 0:
            valid_mask = [1.0] * 6

        # Directional sensors
        sensors = [-1.0] * 6
        for n in neighbors:
            n_info = nodes_info[n]
            d      = np.array([
                n_info['latitude']  - c_info['latitude'],
                n_info['longitude'] - c_info['longitude'],
                n_info['floor']     - c_info['floor'],
            ])
            d_norm         = d / (np.linalg.norm(d) + 1e-9)
            best_d, best_v = -1, 0.5
            for idx, v in dirs:
                dot = np.dot(d_norm, v)
                if abs(v[2]) > 0:
                    dot = 2.0 if (np.sign(d[2]) == np.sign(v[2]) and abs(d[2]) > 0) else -1.0
                if dot > best_v:
                    best_v, best_d = dot, idx
            if best_d != -1:
                nd = np.linalg.norm([
                    n_info['latitude']  - t_info['latitude'],
                    n_info['longitude'] - t_info['longitude'],
                ])
                sensors[best_d] = 1.0 if nd < dist_now else -0.5

        gps = [
            (t_info['latitude']  - c_info['latitude'])  / 200.0,
            (t_info['longitude'] - c_info['longitude']) / 200.0,
            (t_info['floor']     - c_info['floor'])     / 5.0,
        ]
        state = np.concatenate([sensors, gps, node_sigs[curr_id], node_sigs[targ_id]])

        state_t  = torch.FloatTensor(state).to(DEVICE)
        mask_t   = torch.FloatTensor(valid_mask).to(DEVICE)
        features = agent._get_features(state_t)
        logits   = agent.actor(features)

        # Tabu penalty
        for a_idx, (_, v_vec) in enumerate(dirs):
            if valid_mask[a_idx] > 0.0:
                act_vec      = dirs[a_idx][1]
                best_s, best_n_look = -99, None
                for n in neighbors:
                    n_info = nodes_info[n]
                    d = np.array([
                        n_info['latitude']  - c_info['latitude'],
                        n_info['longitude'] - c_info['longitude'],
                        n_info['floor']     - c_info['floor'],
                    ])
                    sim = np.dot(d / (np.linalg.norm(d) + 1e-9), act_vec)
                    if abs(act_vec[2]) > 0 and np.sign(d[2]) == np.sign(act_vec[2]):
                        sim = 2.0
                    if sim > best_s:
                        best_s, best_n_look = sim, n
                if best_n_look in visited[-2:]:
                    logits[a_idx] -= 10.0

        logits = logits + (mask_t - 1.0) * 1e9
        action = torch.argmax(logits).item()

        act_vec        = dirs[action][1]
        best_n, best_s = None, -99
        for n in neighbors:
            n_info = nodes_info[n]
            d = np.array([
                n_info['latitude']  - c_info['latitude'],
                n_info['longitude'] - c_info['longitude'],
                n_info['floor']     - c_info['floor'],
            ])
            sim = np.dot(d / (np.linalg.norm(d) + 1e-9), act_vec)
            if abs(act_vec[2]) > 0 and np.sign(d[2]) == np.sign(act_vec[2]):
                sim = 2.0
            if sim > best_s:
                best_s, best_n = sim, n

        if best_n is not None and best_s > 0.0:
            curr_id = best_n
            agent_path.append(curr_id)
            visited.append(curr_id)
            if curr_id == targ_id:
                done = True
        else:
            done = True

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

reached   = curr_id == targ_id
path_eff  = (true_hop_dist / total_steps * 100) if reached else 0.0

visited_landmarks   = [lm for lm in landmarks if lm in agent_path]
unvisited_landmarks = [lm for lm in landmarks if lm not in agent_path]

print(f"[RESULT] Reached goal: {reached}")
print(f"[RESULT] Steps taken : {total_steps} (optimal: {true_hop_dist})")
print(f"[RESULT] Path efficiency: {path_eff:.1f}%")
print(f"[RESULT] Passed through {len(visited_landmarks)}/{NUM_LANDMARKS} landmarks en route")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

print("\n[INFO] Generating floor trajectory plot...")
plt.figure(figsize=(12, 10), dpi=150)
plt.grid(color='#f0f0f0', linestyle='-', linewidth=1, zorder=0)

# All nodes (grey dots — floor plan)
all_x = [nodes_info[nid]['longitude'] for nid in valid_ids]
all_y = [nodes_info[nid]['latitude']  for nid in valid_ids]
plt.scatter(all_x, all_y, c='#d3d3d3', s=15, alpha=0.6, edgecolors='none', label='Floor Plan Nodes')

# BFS shortest path (solid black line)
tx = [nodes_info[nid]['longitude'] for nid in shortest_path]
ty = [nodes_info[nid]['latitude']  for nid in shortest_path]
plt.plot(tx, ty, color='black', linewidth=3, label='BFS Shortest Path', zorder=2)
plt.scatter(tx, ty, color='black', s=40, zorder=3)

# Agent path (red X markers)
ax_ = [nodes_info[nid]['longitude'] for nid in agent_path]
ay_ = [nodes_info[nid]['latitude']  for nid in agent_path]
plt.scatter(ax_, ay_, color='red', marker='x', s=80, linewidth=2,
            label='OQA-PPO Navigation', zorder=4)

# Unvisited landmarks (pale yellow pentagons)
ux = [nodes_info[nid]['longitude'] for nid in unvisited_landmarks]
uy = [nodes_info[nid]['latitude']  for nid in unvisited_landmarks]
if ux:
    plt.scatter(ux, uy, color='#ffeebf', marker='p', s=200,
                edgecolors='gray', linewidths=1.5,
                label='Off-Route Waypoint', zorder=5)

# Visited landmarks (bright gold pentagons)
vx = [nodes_info[nid]['longitude'] for nid in visited_landmarks]
vy = [nodes_info[nid]['latitude']  for nid in visited_landmarks]
if vx:
    plt.scatter(vx, vy, color='#ffcc00', marker='p', s=300,
                edgecolors='black', linewidths=2,
                label='Passed Waypoint', zorder=6)

# Start and goal
plt.scatter(nodes_info[start_id]['longitude'], nodes_info[start_id]['latitude'],
            color='#2ca02c', s=300, edgecolors='black', zorder=7)
plt.scatter(nodes_info[targ_id]['longitude'], nodes_info[targ_id]['latitude'],
            color='#d62728', s=300, edgecolors='black', zorder=7)
plt.text(nodes_info[start_id]['longitude'] + 3, nodes_info[start_id]['latitude'] + 3,
         "START", fontweight='bold',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
plt.text(nodes_info[targ_id]['longitude'] + 3, nodes_info[targ_id]['latitude'] + 3,
         "GOAL", fontweight='bold',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

status_str = (
    f"{'✓ Reached' if reached else '✗ Failed'} | "
    f"Steps: {total_steps} / Optimal: {true_hop_dist} | "
    f"Efficiency: {path_eff:.1f}%"
)
plt.title(
    f"OQA-PPO Navigation Trajectory\n{status_str}",
    fontsize=14, fontweight='bold', pad=20
)
plt.xlabel("Longitude (m)")
plt.ylabel("Latitude (m)")
plt.legend(loc='upper right', frameon=True, shadow=True)
plt.tight_layout()

out_path = 'ppo_trajectory.pdf'
plt.savefig(out_path, bbox_inches='tight')
print(f"[SUCCESS] Trajectory plot saved: {out_path}")