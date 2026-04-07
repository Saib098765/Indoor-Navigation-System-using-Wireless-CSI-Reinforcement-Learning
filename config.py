import os

PATHS = {
    'data_processed': 'uji_data',
    'models': 'models',
    'results': 'results',
}

TRAINING_CONFIG = {
    'device': 'cpu',
}

PPO_CONFIG = {
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'eps_clip': 0.2,
    'k_epochs': 4,
    'batch_size': 64,
    'max_grad_norm': 0.5,
}

REWARD_CONFIG = {
    'goal_reward': 1000.0,
    'step_penalty': -1.0, 
    'progress_reward': 50.0,
    'regress_penalty': -20.0,
    'invalid_action_penalty': -100.0,
    'timeout_penalty': -500.0, 
}
