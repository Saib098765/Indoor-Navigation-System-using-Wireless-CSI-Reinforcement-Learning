"""
generate_plot.py
================
Zero-shot generalization benchmark for the OQA-PPO agent on the
synthetic 25x25 grid campus (625 nodes, max diameter 48 hops).

Tests success rate and path efficiency across path lengths 1-36 hops.
The agent is evaluated greedily (argmax policy, no sampling noise).

Key fix vs. previous version:
  - BFS now uses a proper node_to_idx dict instead of valid_ids.index()
    which was O(N) per call -> O(N^2) total. Now O(1) per call -> O(N).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm
from train_part2_hierarchical import HybridPPOAgent, DEVICE

# --- Point to whichever checkpoint you want to evaluate ---
# Use best_oqa_ppo_agent.pth for the best mid-training checkpoint,
# or final_oqa_ppo_agent.pth for the end-of-training weights.
MODEL_PATH = "models/final_hybrid_fappo_agent.pth"
DATA_DIR   = "data/processed"


# ---------------------------------------------------------------------------
# Environment loader
# ---------------------------------------------------------------------------

def load_synthetic_env():
    print("[INFO] Loading Synthetic Grid Campus...")

    with open(f"{DATA_DIR}/location_info_synthetic.pkl", 'rb') as f:
        locs = pickle.load(f)
    nodes = {l['location_id']: l for l in locs}

    with open(f"{DATA_DIR}/topology_graph_synthetic.pkl", 'rb') as f:
        G = pickle.load(f)

    valid_ids   = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(valid_ids)}   # O(1) lookup
    adj         = {n: list(G.neighbors(n)) for n in valid_ids}

    num_nodes   = len(valid_ids)
    dist_matrix = np.full((num_nodes, num_nodes), 999, dtype=np.int32)

    print("[INFO] Computing BFS distance matrix (O(N) per node)...")
    for start_node in tqdm(valid_ids, desc="BFS"):
        s_i   = node_to_idx[start_node]
        dist_matrix[s_i, s_i] = 0
        queue   = [(start_node, 0)]
        visited = {start_node}

        while queue:
            curr, d = queue.pop(0)
            if d > 48:                                      # grid diameter = 48
                continue
            dist_matrix[s_i, node_to_idx[curr]] = d
            for n in adj[curr]:
                if n not in visited:
                    visited.add(n)
                    queue.append((n, d + 1))

    # CSI signatures are zero on synthetic campus (no real WiFi data)
    node_sigs = {nid: np.zeros(128) for nid in valid_ids}

    return valid_ids, nodes, adj, dist_matrix, node_sigs


# ---------------------------------------------------------------------------
# Greedy navigation helper
# ---------------------------------------------------------------------------

def navigate_greedy(agent, curr_id, targ_id, nodes_info, adj, node_sigs,
                    dirs, max_steps: int):
    """
    Run one episode greedily (argmax logits).
    Returns (success: bool, steps_taken: int).
    """
    visited = []
    for step in range(1, max_steps + 1):
        c_info = nodes_info[curr_id]
        t_info = nodes_info[targ_id]
        neighbors  = adj[curr_id]
        dist_to_goal = np.linalg.norm([
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
                sensors[best_d] = 1.0 if nd < dist_to_goal else -0.5

        gps = [
            (t_info['latitude']  - c_info['latitude'])  / 200.0,
            (t_info['longitude'] - c_info['longitude']) / 200.0,
            (t_info['floor']     - c_info['floor'])     / 5.0,
        ]
        state  = np.concatenate([sensors, gps, node_sigs[curr_id], node_sigs[targ_id]])

        state_t = torch.FloatTensor(state).to(DEVICE)
        mask_t  = torch.FloatTensor(valid_mask).to(DEVICE)
        features = agent._get_features(state_t)
        logits   = agent.actor(features)

        # Tabu penalty: discourage revisiting recent nodes
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
            visited.append(curr_id)
            if curr_id == targ_id:
                return True, step
        else:
            return False, step   # blocked, no valid move

    return False, max_steps


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

valid_ids, nodes_info, adj, dist_matrix, node_sigs = load_synthetic_env()
node_to_idx = {n: i for i, n in enumerate(valid_ids)}

agent = HybridPPOAgent(128).to(DEVICE)
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. "
        "Run train_part2_hierarchical.py first."
    )
agent.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
agent.eval()

print(f"\n[INFO] Model loaded: {MODEL_PATH}")
print("[INFO] Running zero-shot generalization benchmark (1-36 hops)...\n")

dirs        = [
    (0, [1, 0, 0]), (1, [-1, 0, 0]),
    (2, [0, 1, 0]), (3, [0, -1, 0]),
    (4, [0, 0, 1]), (5, [0, 0, -1]),
]
test_levels = list(range(1, 37))
ATTEMPTS    = 50   # episodes per hop level

success_rates    = []
efficiency_rates = []

with torch.no_grad():
    for level in test_levels:
        successes        = 0
        total_efficiency = 0.0
        valid_attempts   = 0

        for _ in range(ATTEMPTS):
            s_idx    = np.random.randint(len(valid_ids))
            possible = np.where(dist_matrix[s_idx] == level)[0]
            if len(possible) == 0:
                continue

            t_idx    = np.random.choice(possible)
            curr_id  = valid_ids[s_idx]
            targ_id  = valid_ids[t_idx]
            max_steps = int(level * 1.5) + 5
            valid_attempts += 1

            success, steps = navigate_greedy(
                agent, curr_id, targ_id,
                nodes_info, adj, node_sigs,
                dirs, max_steps
            )
            if success:
                successes        += 1
                total_efficiency += level / steps   # 1.0 = optimal path taken

        rate = (successes / max(1, valid_attempts)) * 100
        eff  = (total_efficiency / max(1, successes)) * 100 if successes > 0 else 0.0
        success_rates.append(rate)
        efficiency_rates.append(eff)

        print(
            f"Hops: {level:2d} | "
            f"Success: {rate:5.1f}% | "
            f"Efficiency: {eff:5.1f}% | "
            f"({successes}/{valid_attempts})"
        )

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax1 = plt.subplots(figsize=(11, 6))

color1 = '#2ca02c'
color2 = '#1f77b4'

ax1.set_xlabel('Target Distance (Hops)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Success Rate (%)', color=color1, fontsize=12, fontweight='bold')
line1 = ax1.plot(
    test_levels, success_rates,
    marker='o', color=color1, linewidth=2.5, markersize=7, label='Success Rate'
)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_ylim(-5, 105)
ax1.set_xticks(np.arange(0, max(test_levels) + 1, 2))

ax2 = ax1.twinx()
ax2.set_ylabel('Path Efficiency (%)', color=color2, fontsize=12, fontweight='bold')
line2 = ax2.plot(
    test_levels, efficiency_rates,
    marker='s', linestyle='--', color=color2,
    linewidth=2, markersize=6, label='Routing Efficiency'
)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(-5, 105)

lines  = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower left', frameon=True, shadow=True)

plt.title(
    'OQA-PPO Zero-Shot Generalization Benchmark\n'
    '(Synthetic 25×25 Grid Campus — Agent trained up to curriculum level)',
    fontsize=13, fontweight='bold', pad=15
)
plt.tight_layout()
out_path = 'ppo_generalization_plot.pdf'
plt.savefig(out_path, format='pdf', dpi=300)
print(f"\n[INFO] Plot saved: {out_path}")