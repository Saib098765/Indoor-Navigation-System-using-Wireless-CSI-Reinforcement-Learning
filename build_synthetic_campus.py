"""
build_synthetic_campus.py
=========================
Generates a synthetic 25x25 grid campus (625 nodes, 1200 edges) for
zero-shot generalization benchmarking of the OQA-PPO agent.

NOTE: This is NOT a subset of the real UJI dataset. It is a controlled
synthetic environment used to benchmark how well the agent generalizes
to path lengths it was NOT trained on (up to 36-hop paths on a grid
with diameter 48 hops).

Why a grid campus?
  - Clean topology: no dead ends, uniform connectivity
  - Known ground truth: BFS distance is exact
  - Reproducible: same graph every run
  - Stress test: forces agent to navigate long corridors without shortcuts

Output files (in data/processed/):
  - topology_graph_synthetic.pkl    : networkx Graph
  - location_info_synthetic.pkl     : list of node dicts with lat/lon/floor
"""

import networkx as nx
import numpy as np
import pickle
import os

DATA_DIR = "data/processed"
os.makedirs(DATA_DIR, exist_ok=True)

GRID_SIZE   = 25          # 25x25 = 625 nodes
SPACING_M   = 10.0        # 10 metres between adjacent nodes
FLOOR_NUM   = 1           # Single floor (all nodes same floor)

print(f"\n[INFO] Generating {GRID_SIZE}x{GRID_SIZE} Grid Campus...")
print(f"       Nodes   : {GRID_SIZE * GRID_SIZE}")
print(f"       Spacing : {SPACING_M}m between nodes")
print(f"       Max diameter : {2 * (GRID_SIZE - 1)} hops\n")

# --- 1. Build grid graph ---
G = nx.grid_2d_graph(GRID_SIZE, GRID_SIZE)

# --- 2. Convert (row, col) coordinates to integer node IDs ---
node_to_id = {n: i for i, n in enumerate(G.nodes())}
G_export   = nx.Graph()
locations  = []

for n in G.nodes():
    nid = node_to_id[n]
    G_export.add_node(nid)
    locations.append({
        'location_id': nid,
        'latitude':    n[0] * SPACING_M,   # row -> latitude (metres)
        'longitude':   n[1] * SPACING_M,   # col -> longitude (metres)
        'floor':       FLOOR_NUM,
    })

for u, v in G.edges():
    G_export.add_edge(node_to_id[u], node_to_id[v])

# --- 3. Validate ---
assert len(G_export.nodes()) == GRID_SIZE * GRID_SIZE, "Node count mismatch"
assert nx.is_connected(G_export),                       "Graph is not connected"

# --- 4. Save ---
graph_path = f"{DATA_DIR}/topology_graph_synthetic.pkl"
locs_path  = f"{DATA_DIR}/location_info_synthetic.pkl"

with open(graph_path, 'wb') as f:
    pickle.dump(G_export, f)

with open(locs_path, 'wb') as f:
    pickle.dump(locations, f)

print(f"[SUCCESS] Synthetic campus saved.")
print(f"          Graph  -> {graph_path}")
print(f"          Locs   -> {locs_path}")
print(f"\n  Nodes : {len(G_export.nodes())}")
print(f"  Edges : {len(G_export.edges())}")
print(f"  Max graph diameter (hops) : {2 * (GRID_SIZE - 1)}")
print(f"\n  NOTE: CSI signatures are zero-vectors on this campus.")
print(f"        UW-GAE omega will be ~1.0 (no down-weighting) since")
print(f"        var(zeros) = 0. This is expected and correct behaviour.")
print(f"        FA-PPO epsilon will stay at eps_base for the same reason.")
print(f"        The synthetic campus tests generalization, not OQA mechanisms.")