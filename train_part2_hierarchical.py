"""
OQA-PPO: Observation-Quality-Aware Proximal Policy Optimization
================================================================
Two-part novel contribution for CSI-based indoor navigation:

  1. FA-PPO (Fading-Aware PPO):
       Dynamic clip epsilon based on CSI spectral dispersion.
       epsilon_t = max(0.05, eps_base * exp(-lambda * sigma^2_CSI))
       High CSI variance => smaller trust region => cautious policy updates.

  2. UW-GAE (Uncertainty-Weighted GAE):
       Advantage estimates are discounted when the observation is unreliable.
       A_t^{UW} = omega_t * A_t^{GAE},  omega_t = exp(-mu * sigma^2_CSI)
       High CSI variance => low credit given to that transition's gradient signal.

Both mechanisms share a single CSI variance computation per update step.
"""

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

from config import PPO_CONFIG, REWARD_CONFIG

DATA_DIR = "uji_data"
MODEL_DIR = "models"
DEVICE = torch.device('cpu')

try:
    from train_part1_hierarchical_v2 import HierarchicalClassifier
    HAS_WIFI = True
except ImportError:
    HAS_WIFI = False


# ---------------------------------------------------------------------------
# Environment loading helpers
# ---------------------------------------------------------------------------

def get_signatures():
    """Extract per-node CSI fingerprint embeddings from the trained classifier."""
    if not HAS_WIFI:
        return None
    clf_path = f"{MODEL_DIR}/hierarchical_classifier_best.pth"
    if not os.path.exists(clf_path):
        return None

    try:
        ckpt = torch.load(clf_path, map_location=DEVICE)
        model = HierarchicalClassifier(
            128, ckpt['num_buildings'], ckpt['num_floors'], ckpt['num_rooms']
        ).to(DEVICE)
        model.load_state_dict(ckpt['model_state_dict'])

        df = pd.read_csv(f"{DATA_DIR}/uji_train.csv")
        with open(f"{MODEL_DIR}/scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
        X = scaler.transform(df[[f'CSI_DATA{i}' for i in range(1, 129)]].values)
        y = df['label'].values

        dataset = TensorDataset(torch.FloatTensor(X.copy()), torch.LongTensor(y))
        loader = DataLoader(dataset, batch_size=256, shuffle=False)

        sigs_sum, sigs_cnt = {}, {}
        model.eval()
        with torch.no_grad():
            for Xb, yb in loader:
                feats = model.features(Xb.to(DEVICE)).cpu().numpy()
                for i, lbl in enumerate(yb.numpy()):
                    if lbl not in sigs_sum:
                        sigs_sum[lbl] = np.zeros_like(feats[i])
                        sigs_cnt[lbl] = 0
                    sigs_sum[lbl] += feats[i]
                    sigs_cnt[lbl] += 1
        return {k: v / sigs_cnt[k] for k, v in sigs_sum.items()}
    except Exception:
        return None


def load_aligned_env():
    """Load building graph, compute BFS distance matrix, and attach CSI signatures."""
    path_loc = f"{DATA_DIR}/location_info_building0.pkl"
    with open(path_loc, 'rb') as f:
        locs = pickle.load(f)
    nodes = {l['location_id']: l for l in locs if abs(l.get('latitude', 0)) > 1.0}

    path_graph = f"{DATA_DIR}/topology_graph_building0.pkl"
    if not os.path.exists(path_graph):
        print("Error: topology_graph_building0.pkl missing. Run rebuild_graph.py first.")
        exit()

    with open(path_graph, 'rb') as f:
        G = pickle.load(f)

    valid_ids = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(valid_ids)}

    num_nodes = len(valid_ids)
    dist_matrix = np.full((num_nodes, num_nodes), 999)

    adj = {n: list(G.neighbors(n)) for n in valid_ids}
    for start_node in tqdm(valid_ids, desc="BFS distance matrix"):
        s_i = node_to_idx[start_node]
        dist_matrix[s_i, s_i] = 0
        queue = [(start_node, 0)]
        visited = {start_node}

        while queue:
            curr, d = queue.pop(0)
            if d > 20:
                continue
            dist_matrix[s_i, node_to_idx[curr]] = d
            for n in adj[curr]:
                if n not in visited:
                    visited.add(n)
                    queue.append((n, d + 1))

    sigs = get_signatures()
    node_sigs = {}
    if sigs:
        for nid in valid_ids:
            s = sigs.get(nid, np.zeros(128))
            node_sigs[nid] = s / (np.linalg.norm(s) + 1e-9)
    else:
        for nid in valid_ids:
            node_sigs[nid] = np.zeros(128)

    return valid_ids, nodes, adj, dist_matrix, node_sigs


# ---------------------------------------------------------------------------
# Neural network: Dual-Stream Sensor Fusion Actor-Critic
# ---------------------------------------------------------------------------

class HybridPPOAgent(nn.Module):
    """
    Dual-stream architecture:
      Stream 1 — CSI extractor: compresses (current + target) 256-dim CSI pair -> 16-dim
      Stream 2 — Geometry:      6 directional sensors + 3 GPS deltas -> 9-dim
    Fused -> shared trunk -> actor head + critic head
    """

    def __init__(self, wifi_dim: int):
        super().__init__()

        # Stream 1: CSI fingerprint compression
        self.csi_extractor = nn.Sequential(
            nn.Linear(wifi_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
        )

        # Shared trunk (geometry 9-dim + CSI features 16-dim = 25-dim input)
        self.shared = nn.Sequential(
            nn.Linear(9 + 16, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        # Policy head
        self.actor = nn.Sequential(nn.Linear(64, 6))

        # Value head
        self.critic = nn.Sequential(nn.Linear(64, 1))

        self.opt = optim.Adam(self.parameters(), lr=PPO_CONFIG['learning_rate'])

    def _get_features(self, state_t: torch.Tensor) -> torch.Tensor:
        is_batch = state_t.dim() == 2
        geometry = state_t[:, :9]   if is_batch else state_t[:9]
        csi_data = state_t[:, 9:]   if is_batch else state_t[9:]

        csi_features = self.csi_extractor(csi_data)

        fused = (
            torch.cat([geometry, csi_features], dim=1) if is_batch
            else torch.cat([geometry, csi_features], dim=0)
        )
        return self.shared(fused)

    def act(self, state: np.ndarray, mask: list):
        """Sample an action stochastically (used during training rollouts)."""
        state_t = torch.FloatTensor(state).to(DEVICE)
        mask_t  = torch.FloatTensor(mask).to(DEVICE)
        features = self._get_features(state_t)

        logits = self.actor(features)
        logits = logits + (mask_t - 1.0) * 1e9   # mask invalid actions

        probs = torch.clamp(torch.softmax(logits, dim=-1), min=1e-6)
        dist  = Categorical(probs)
        action = dist.sample()
        value  = self.critic(features)

        return action.item(), dist.log_prob(action), value.squeeze(-1)

    def evaluate(self, states: torch.Tensor, actions: torch.Tensor, masks: torch.Tensor):
        """Re-evaluate a batch of (state, action) pairs (used during PPO update)."""
        features = self._get_features(states)
        logits   = self.actor(features)
        logits   = logits + (masks - 1.0) * 1e9

        probs = torch.clamp(torch.softmax(logits, dim=-1), min=1e-6)
        dist  = Categorical(probs)

        return dist.log_prob(actions), self.critic(features).squeeze(-1), dist.entropy()


# ---------------------------------------------------------------------------
# OQA-PPO Update: FA-PPO + UW-GAE
# ---------------------------------------------------------------------------

def ppo_update(agent: HybridPPOAgent, memory: dict) -> dict:
    """
    Full OQA-PPO update combining:
      - FA-PPO:  per-sample dynamic clip epsilon (shape [N], no unsqueeze bug)
      - UW-GAE:  per-sample uncertainty-weighted advantages
      - Mini-batch k-epoch optimization
    Returns a dict of scalar loss metrics for logging.
    """
    old_states    = torch.FloatTensor(np.array(memory['states'])).to(DEVICE)
    old_actions   = torch.LongTensor(memory['actions']).to(DEVICE)
    old_logprobs  = torch.stack(memory['logprobs']).to(DEVICE).detach()
    old_masks     = torch.FloatTensor(np.array(memory['masks'])).to(DEVICE)
    rewards       = memory['rewards']
    values        = memory['values']
    dones         = memory['dones']

    # ------------------------------------------------------------------
    # STEP 1 — CSI variance: shared signal quality proxy for both mechanisms
    # Shape: [N]  — one scalar per transition
    # dim=1 computes variance across the 128 CSI feature dimensions (spectral
    # dispersion). This is a valid proxy for signal quality even though true
    # fading is temporal; for a static fingerprint database this is the
    # best available signal-quality signal. Acknowledge this in your thesis.
    # ------------------------------------------------------------------
    csi_data     = old_states[:, 9:137]                           # shape [N, 128]
    csi_variance = torch.var(csi_data, dim=1) + 1e-6             # shape [N]

    # ------------------------------------------------------------------
    # STEP 2 — FA-PPO: Dynamic epsilon (no unsqueeze — must stay [N])
    # epsilon_t = max(0.05, eps_base * exp(-lambda * sigma^2_CSI))
    # ------------------------------------------------------------------
    lambda_coef = PPO_CONFIG['lambda_coef']
    dynamic_eps = PPO_CONFIG['eps_clip'] * torch.exp(-lambda_coef * csi_variance)
    dynamic_eps = torch.clamp(dynamic_eps, min=0.05)              # shape [N]

    # ------------------------------------------------------------------
    # STEP 3 — GAE: Generalized Advantage Estimation
    # ------------------------------------------------------------------
    returns, advantages, gae = [], [], 0.0
    for t in reversed(range(len(rewards))):
        next_val = 0 if (t == len(rewards) - 1 or dones[t]) else values[t + 1].item()
        delta    = rewards[t] + PPO_CONFIG['gamma'] * next_val - values[t].item()
        gae      = delta + PPO_CONFIG['gamma'] * PPO_CONFIG['gae_lambda'] * gae * (1 - int(dones[t]))
        advantages.insert(0, gae)
        returns.insert(0, gae + values[t].item())

    returns    = torch.FloatTensor(returns).to(DEVICE)
    advantages = torch.FloatTensor(advantages).to(DEVICE)

    # ------------------------------------------------------------------
    # STEP 4 — UW-GAE: Uncertainty-Weighted Advantage Estimation
    # A_t^{UW} = omega_t * A_t^{GAE},  omega_t = exp(-mu * sigma^2_CSI)
    # High variance => omega close to 0 => noisy transitions contribute
    # little to the policy gradient. Normalize AFTER weighting.
    # ------------------------------------------------------------------
    mu_coef = PPO_CONFIG['mu_coef']
    omega   = torch.exp(-mu_coef * csi_variance)                 # shape [N], in (0,1]
    advantages = advantages * omega                               # UW-GAE applied here

    if len(advantages) > 1 and advantages.std() > 1e-5:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    # ------------------------------------------------------------------
    # STEP 5 — Mini-batch k-epoch PPO optimization
    # ------------------------------------------------------------------
    dataset_size = len(memory['states'])
    batch_size   = PPO_CONFIG.get('batch_size', 64)

    total_actor_loss, total_critic_loss, total_entropy = 0.0, 0.0, 0.0
    num_updates = 0

    for _ in range(PPO_CONFIG['k_epochs']):
        indices = torch.randperm(dataset_size).to(DEVICE)

        for start in range(0, dataset_size, batch_size):
            mb_idx = indices[start:start + batch_size]

            mb_states      = old_states[mb_idx]
            mb_actions     = old_actions[mb_idx]
            mb_old_logprobs = old_logprobs[mb_idx]
            mb_masks       = old_masks[mb_idx]
            mb_advantages  = advantages[mb_idx]
            mb_returns     = returns[mb_idx]
            mb_dynamic_eps = dynamic_eps[mb_idx]               # shape [mb_size]

            logprobs, state_values, dist_entropy = agent.evaluate(
                mb_states, mb_actions, mb_masks
            )

            ratios = torch.exp(logprobs - mb_old_logprobs)    # shape [mb_size]

            # FA-PPO clipped surrogate — mb_dynamic_eps is [mb_size], ratios is [mb_size]
            # No broadcasting issue: shapes match exactly.
            surr1      = ratios * mb_advantages
            surr2      = torch.clamp(ratios, 1 - mb_dynamic_eps, 1 + mb_dynamic_eps) * mb_advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = nn.SmoothL1Loss()(state_values, mb_returns)
            entropy     = dist_entropy.mean()

            loss = (
                actor_loss
                + PPO_CONFIG['value_coef']   * critic_loss
                - PPO_CONFIG['entropy_coef'] * entropy
            )

            agent.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), PPO_CONFIG['max_grad_norm'])
            agent.opt.step()

            total_actor_loss  += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_entropy     += entropy.item()
            num_updates       += 1

    return {
        'actor_loss':  total_actor_loss  / max(1, num_updates),
        'critic_loss': total_critic_loss / max(1, num_updates),
        'entropy':     total_entropy     / max(1, num_updates),
        'mean_omega':  omega.mean().item(),       # how much UW-GAE down-weighted on avg
        'mean_eps':    dynamic_eps.mean().item(), # how much FA-PPO shrunk epsilon on avg
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)

    valid_ids, nodes_info, adj, dist_matrix, node_sigs = load_aligned_env()
    agent = HybridPPOAgent(128).to(DEVICE)

    print("\nStarting OQA-PPO Training (FA-PPO + UW-GAE, Mini-Batch Optimized)...")
    print(f"  lambda_coef (FA-PPO) = {PPO_CONFIG['lambda_coef']}")
    print(f"  mu_coef     (UW-GAE) = {PPO_CONFIG['mu_coef']}\n")

    dirs = [
        (0, [1, 0, 0]), (1, [-1, 0, 0]),
        (2, [0, 1, 0]), (3, [0, -1, 0]),
        (4, [0, 0, 1]), (5, [0, 0, -1]),
    ]

    curriculum_level = 1
    success_buffer   = []
    best_rate        = 0.0
    pbar             = tqdm(range(50000))

    memory = {k: [] for k in ('states', 'actions', 'logprobs', 'values', 'rewards', 'dones', 'masks')}

    # Metrics log (printed every 50 episodes, useful for your thesis plots)
    log = {
        'episode': [], 'success_rate': [], 'curriculum_level': [],
        'actor_loss': [], 'critic_loss': [], 'entropy': [],
        'mean_omega': [], 'mean_eps': [],
    }

    for ep in pbar:
        s_idx    = np.random.randint(len(valid_ids))
        possible = np.where(dist_matrix[s_idx] == curriculum_level)[0]
        if len(possible) == 0:
            possible = np.where(
                (dist_matrix[s_idx] <= curriculum_level) & (dist_matrix[s_idx] > 0)
            )[0]
            if len(possible) == 0:
                continue

        t_idx          = np.random.choice(possible)
        curr_id        = valid_ids[s_idx]
        targ_id        = valid_ids[t_idx]
        c_info, t_info = nodes_info[curr_id], nodes_info[targ_id]

        done               = False
        steps              = 0
        true_distance_hops = dist_matrix[s_idx, t_idx]
        max_steps          = int(true_distance_hops * 1.5) + 5
        dist_to_goal       = np.linalg.norm([
            c_info['latitude']  - t_info['latitude'],
            c_info['longitude'] - t_info['longitude'],
        ])

        while not done and steps < max_steps:
            steps += 1
            sensors  = [-1.0] * 6
            c_info   = nodes_info[curr_id]
            neighbors = adj[curr_id]

            # Build action mask
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

            # Fill directional sensors
            for n in neighbors:
                n_info = nodes_info[n]
                d      = np.array([
                    n_info['latitude']  - c_info['latitude'],
                    n_info['longitude'] - c_info['longitude'],
                    n_info['floor']     - c_info['floor'],
                ])
                d_norm           = d / (np.linalg.norm(d) + 1e-9)
                best_d, best_v   = -1, 0.5
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

            state           = np.concatenate([sensors, gps, node_sigs[curr_id], node_sigs[targ_id]])
            action, lp, value = agent.act(state, valid_mask)

            # Execute action
            act_vec        = dirs[action][1]
            best_n, best_s = None, -99
            for n in neighbors:
                n_info = nodes_info[n]
                d      = np.array([
                    n_info['latitude']  - c_info['latitude'],
                    n_info['longitude'] - c_info['longitude'],
                    n_info['floor']     - c_info['floor'],
                ])
                sim = np.dot(d / (np.linalg.norm(d) + 1e-9), act_vec)
                if abs(act_vec[2]) > 0 and np.sign(d[2]) == np.sign(act_vec[2]):
                    sim = 2.0
                if sim > best_s:
                    best_s, best_n = sim, n

            r = REWARD_CONFIG['step_penalty']
            if best_n is not None and best_s > 0.0:
                curr_id  = best_n
                c_info   = nodes_info[curr_id]
                new_d    = np.linalg.norm([
                    c_info['latitude']  - t_info['latitude'],
                    c_info['longitude'] - t_info['longitude'],
                ])
                r += np.clip(dist_to_goal - new_d, -1.0, 1.0) * (REWARD_CONFIG['progress_reward'] / 10.0)
                dist_to_goal = new_d

                if curr_id == targ_id:
                    r   += REWARD_CONFIG['goal_reward']
                    done = True
            else:
                r += REWARD_CONFIG['invalid_action_penalty']

            if not done and steps >= max_steps:
                r   += REWARD_CONFIG['timeout_penalty']
                done = True

            memory['states'].append(state)
            memory['actions'].append(action)
            memory['logprobs'].append(lp)
            memory['values'].append(value)
            memory['rewards'].append(r)
            memory['dones'].append(done)
            memory['masks'].append(valid_mask)

        success_buffer.append(1 if curr_id == targ_id else 0)

        # PPO update every 2000 steps
        last_metrics = {}
        if len(memory['states']) >= 2000:
            last_metrics = ppo_update(agent, memory)
            memory = {k: [] for k in memory}

        # Logging every 50 episodes
        if (ep + 1) % 50 == 0:
            rate = np.mean(success_buffer[-50:])
            if rate > best_rate:
                best_rate = rate
                torch.save(agent.state_dict(), f"{MODEL_DIR}/final_hybrid_fappo_agent.pth")

            desc = f"Lvl {curriculum_level} | Succ: {rate:.1%} | Best: {best_rate:.1%}"
            if last_metrics:
                desc += (
                    f" | ε={last_metrics['mean_eps']:.3f}"
                    f" | ω={last_metrics['mean_omega']:.3f}"
                )
            pbar.set_description(desc)

            if last_metrics:
                log['episode'].append(ep + 1)
                log['success_rate'].append(rate)
                log['curriculum_level'].append(curriculum_level)
                log['actor_loss'].append(last_metrics['actor_loss'])
                log['critic_loss'].append(last_metrics['critic_loss'])
                log['entropy'].append(last_metrics['entropy'])
                log['mean_omega'].append(last_metrics['mean_omega'])
                log['mean_eps'].append(last_metrics['mean_eps'])

            if rate > 0.8:
                curriculum_level += 1
                success_buffer   = []

    print(f"\nFinal Curriculum Level Reached: {curriculum_level}")
    print(f"Best Success Rate: {best_rate:.1%}")

    # Save final model
    torch.save(agent.state_dict(), f"{MODEL_DIR}/final_hybrid_fappo_agent.pth")

    # Save training log for thesis plots
    import json
    with open(f"{MODEL_DIR}/training_log.json", 'w') as f:
        json.dump(log, f, indent=2)
    print(f"Training log saved to {MODEL_DIR}/training_log.json")