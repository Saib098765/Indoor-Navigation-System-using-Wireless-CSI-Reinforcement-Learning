import gym
import numpy as np
import networkx as nx
from gym import spaces

class NavigationEnvironment(gym.Env):
    def __init__(self, num_locations, location_signatures, topology, reward_config=None, hub_id=0):
        super(NavigationEnvironment, self).__init__()
        
        self.location_signatures = location_signatures
        self.topology = topology
        self.nodes_data = {n: d for n, d in self.topology.nodes(data=True)}
        # remove 0.0 coordinates
        self.valid_nodes = [n for n, d in self.nodes_data.items() if abs(d.get('latitude',0)) > 1.0]
        
        self.reward_config = reward_config or {
            'step_penalty': -0.05,
            'progress_reward': 1.0,
            'regress_penalty': -0.5,
            'goal_reward': 20.0,
            'invalid_action_penalty': -1.0,
            'timeout_penalty': -5.0
        }
        
        # Actions: 0:N, 1:S, 2:E, 3:W, 4:U, 5:D
        self.action_space = spaces.Discrete(6)
        self.action_space_n = 6

        # State: [Current_Sig (128) + Target_Sig (128) + GPS_Vector (3)]
        first_sig = list(location_signatures.values())[0]
        self.sig_dim = len(first_sig) if isinstance(first_sig, (list, np.ndarray)) else 128
        self.state_dim = (self.sig_dim * 2) + 3
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        
        self.current_location = 0
        self.target_location = 0
        self.step_count = 0
        self.max_steps = 100
        
        try:
            self.shortest_paths = dict(nx.all_pairs_shortest_path_length(self.topology))
        except:
            self.shortest_paths = {}

    def reset(self, start_node=None, target_node=None):
        self.step_count = 0
        if start_node is None:
            self.current_location = np.random.choice(self.valid_nodes)
        else:
            self.current_location = start_node
            
        if target_node is None:
            try:
                reachable = list(nx.node_connected_component(self.topology, self.current_location))
                reachable = [n for n in reachable if n != self.current_location and n in self.valid_nodes]
                if reachable:
                    self.target_location = np.random.choice(reachable)
                else:
                    self.target_location = self.current_location
            except:
                self.target_location = self.current_location
        else:
            self.target_location = target_node
            
        return self._get_state()

    def _get_state(self):
        # 1. Signatures
        curr_sig = self.location_signatures.get(self.current_location, np.zeros(self.sig_dim))
        targ_sig = self.location_signatures.get(self.target_location, np.zeros(self.sig_dim))
        
        # 2. GPS Vector (NORMALIZED)
        curr_info = self.nodes_data.get(self.current_location, {})
        targ_info = self.nodes_data.get(self.target_location, {})
        
        # 3. Divide by 200 mt to keep i/p bw [-1.0 and 1.0]
        d_lat = (targ_info.get('latitude', 0.0) - curr_info.get('latitude', 0.0)) / 200.0
        d_lon = (targ_info.get('longitude', 0.0) - curr_info.get('longitude', 0.0)) / 200.0
        d_floor = float(targ_info.get('floor', 0) - curr_info.get('floor', 0)) / 5.0
        
        vector = np.array([d_lat, d_lon, d_floor], dtype=np.float32)
        vector = np.clip(vector, -1.0, 1.0)
        
        return np.concatenate([curr_sig, targ_sig, vector])

    def step(self, action):
        self.step_count += 1
        done = False
        info = {'arrived': False}
        reward = self.reward_config['step_penalty']
        target_vec = np.zeros(3) 
        if action == 0: target_vec[0] = 1.0   # North
        elif action == 1: target_vec[0] = -1.0 # South
        elif action == 2: target_vec[1] = 1.0  # East
        elif action == 3: target_vec[1] = -1.0 # West
        elif action == 4: target_vec[2] = 1.0  # Up
        elif action == 5: target_vec[2] = -1.0 # Down
        
        curr_info = self.nodes_data.get(self.current_location, {})
        neighbors = list(self.topology.neighbors(self.current_location))
        best_neighbor = None
        best_score = -999.0
        
        for n in neighbors:
            n_info = self.nodes_data.get(n, {})
            d_lat = n_info.get('latitude', 0) - curr_info.get('latitude', 0)
            d_lon = n_info.get('longitude', 0) - curr_info.get('longitude', 0)
            d_floor = float(n_info.get('floor', 0) - curr_info.get('floor', 0))
            mag = np.sqrt(d_lat**2 + d_lon**2 + d_floor**2) + 1e-9
            n_vec = np.array([d_lat, d_lon, d_floor]) / mag
            score = np.dot(n_vec, target_vec)
            
            if abs(target_vec[2]) > 0:
                if np.sign(d_floor) == np.sign(target_vec[2]) and abs(d_floor) > 0: score = 2.0
                else: score = -2.0
            if score > best_score:
                best_score = score
                best_neighbor = n
        
        # 0.1 --> diagonal movement
        if best_neighbor is not None and best_score > 0.1:
            prev_dist = self.shortest_paths.get(self.current_location, {}).get(self.target_location, 999)
            self.current_location = best_neighbor
            curr_dist = self.shortest_paths.get(self.current_location, {}).get(self.target_location, 999)
            
            if self.current_location == self.target_location:
                reward += self.reward_config['goal_reward']
                done = True
                info['arrived'] = True
            elif curr_dist < prev_dist:
                reward += self.reward_config['progress_reward']
            else:
                reward += self.reward_config['regress_penalty']
        else:
            reward += self.reward_config['invalid_action_penalty']
            
        if self.step_count >= self.max_steps:
            done = True
            reward += self.reward_config['timeout_penalty']
            
        return self._get_state(), reward, done, info