import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np

class PPOBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.logprobs = []
        self.is_terminals = []
        self.batch_size = 64

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.logprobs[:]
        del self.is_terminals[:]

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256), 
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_probs = self.actor(state)
        action_probs = torch.clamp(action_probs, min=1e-6)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        action_probs = torch.clamp(action_probs, min=1e-6)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

class PPOAgent:
    def __init__(self, state_dim, action_dim, config, device):
        self.lr = config.get('learning_rate', 0.0003)
        self.gamma = config.get('gamma', 0.99)
        self.eps_clip = config.get('eps_clip', 0.2)
        self.K_epochs = config.get('k_epochs', 4)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.device = device
        self.buffer = PPOBuffer()
        self.buffer.batch_size = config.get('batch_size', 64)
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()

    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            if deterministic:
                action_probs = self.policy_old.actor(state)
                action = torch.argmax(action_probs).item()
                return action, 0, 0
            else:
                action, action_logprob = self.policy_old.act(state)
                return action.item(), action_logprob.item(), 0

    def store_transition(self, state, action, reward, value, log_prob, done):
        self.buffer.states.append(torch.FloatTensor(state))
        self.buffer.actions.append(torch.tensor(action))
        self.buffer.logprobs.append(torch.tensor(log_prob))
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(done)

    def update(self, next_state=None):
        # Monte Carlo Estimate
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        if len(rewards) > 1:
            std = rewards.std()
            if std > 1e-8:
                rewards = (rewards - rewards.mean()) / (std + 1e-8)
            else:
                rewards = rewards - rewards.mean() # Center only if no variance
        
        old_states = torch.stack(self.buffer.states).to(self.device).detach()
        old_actions = torch.stack(self.buffer.actions).to(self.device).detach()
        old_logprobs = torch.stack(self.buffer.logprobs).to(self.device).detach()

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = state_values.view(-1)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            # Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            # Gradient
            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy_old.load_state_dict(checkpoint)
        self.policy.load_state_dict(checkpoint)