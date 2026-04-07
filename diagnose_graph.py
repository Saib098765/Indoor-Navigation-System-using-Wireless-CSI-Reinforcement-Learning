import pickle
import numpy as np
import networkx as nx
import os
from models.navigation_env import NavigationEnvironment

DATA_DIR = "uji_data"

def get_perfect_action(curr_loc, target_loc, nodes_data):
    curr = nodes_data[curr_loc]
    targ = nodes_data[target_loc]
    
    d_lat = targ['latitude'] - curr['latitude']
    d_lon = targ['longitude'] - curr['longitude']
    d_floor = float(targ['floor'] - curr['floor'])
    
    # Priority: Floor -> Lat/Lon
    if abs(d_floor) > 0:
        return 4 if d_floor > 0 else 5
    
    if abs(d_lat) > abs(d_lon):
        return 0 if d_lat > 0 else 1 # North / South
    else:
        return 2 if d_lon > 0 else 3 # East / West

def debug_manual_walk():
    print("\n" + "="*80)
    print("MANUAL WALK DEBUGGER")
    print("="*80)

    try:
        with open(f"{DATA_DIR}/location_info_building0.pkl", 'rb') as f:
            locs = pickle.load(f)
        with open(f"{DATA_DIR}/topology_graph_building0.pkl", 'rb') as f:
            G = pickle.load(f)
    except Exception as e:
        print(f"Error loading: {e}")
        return

    dummy_sigs = {l['location_id']: np.zeros(1) for l in locs}
    
    env = NavigationEnvironment(
        num_locations=len(locs),
        location_signatures=dummy_sigs,
        topology=G
    )
    
    valid_nodes = list(G.nodes())
    start = valid_nodes[0]
    target = None
    
    # BFS to find a node 2 steps away
    for n in G.neighbors(start):
        for nn in G.neighbors(n):
            if nn != start:
                target = nn
                break
        if target: break
        
    print(f"\nTest:")
    print(f"  Start Node: {start}  (Lat: {env.nodes_data[start]['latitude']:.4f}, Lon: {env.nodes_data[start]['longitude']:.4f})")
    print(f"  Target Node: {target} (Lat: {env.nodes_data[target]['latitude']:.4f}, Lon: {env.nodes_data[target]['longitude']:.4f})")
    
    env.reset(start, target)
    done = False
    step = 0
    
    print("\nStarting Walk")
    
    while not done and step < 5:
        step += 1
        print(f"\nSTEP: {step}")
        print(f"  Current: {env.current_location}")
        action = get_perfect_action(env.current_location, env.target_location, env.nodes_data)
        action_str = ["North", "South", "East", "West", "Up", "Down"][action]
        
        print(f"Action: {action} ({action_str})")
        
        curr_info = env.nodes_data[env.current_location]
        neighbors = list(env.topology.neighbors(env.current_location))
        
        target_vec = np.zeros(3)
        if action == 0: target_vec[0] = 1.0
        elif action == 1: target_vec[0] = -1.0
        elif action == 2: target_vec[1] = 1.0
        elif action == 3: target_vec[1] = -1.0
        
        best_score = -99
        best_n = None
        
        for n in neighbors:
            n_info = env.nodes_data[n]
            d_lat = n_info['latitude'] - curr_info['latitude']
            d_lon = n_info['longitude'] - curr_info['longitude']
            mag = np.sqrt(d_lat**2 + d_lon**2) + 1e-9
            vec = np.array([d_lat, d_lon, 0]) / mag
            score = np.dot(vec, target_vec)
            
            print(f"    Neighbor {n}: d_lat={d_lat:.6f}, d_lon={d_lon:.6f} | Score={score:.3f}")
            if score > best_score:
                best_score = score
                best_n = n
        
        print(f"  Best Neighbor: {best_n} with Score {best_score:.3f}")
        
        if best_score > 0.3:
            print("Environment should allow this move.")
        else:
            print("Environment should reject this move (Score < 0.3). Threshold maybe strict")

        _, reward, done, info = env.step(action)
        print(f"  Result: Reward={reward}, Arrived={info['arrived']}")
        
        if info['arrived']:
            print("SUCCESS! Environment works.")
            return

    print("FAILED. Agent did not reach goal.")

if __name__ == "__main__":
    debug_manual_walk()