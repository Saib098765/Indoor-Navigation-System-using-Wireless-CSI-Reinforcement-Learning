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
    'gae_lambda': 0.95,       # Smoothing parameter for GAE
    'eps_clip': 0.2,           # Base clipping parameter (FA-PPO scales this dynamically)
    'k_epochs': 4,
    'batch_size': 64,
    'max_grad_norm': 0.5,
    'value_coef': 0.5,         # Weight for critic loss
    'entropy_coef': 0.01,      # Encourages exploration

    # --- OQA-PPO Hyperparameters ---
    # FA-PPO: Controls how aggressively CSI variance shrinks epsilon.
    #   Higher lambda -> epsilon shrinks faster under high-variance (noisy) CSI.
    #   Range: [0.1, 2.0]. Start at 0.5.
    'lambda_coef': 0.5,

    # UW-GAE: Controls how aggressively CSI variance discounts advantage estimates.
    #   Higher mu -> noisy-CSI transitions contribute less to the policy gradient.
    #   Range: [0.1, 1.0]. Start at 0.3. Keep lower than lambda_coef.
    'mu_coef': 0.3,
}

REWARD_CONFIG = {
    'goal_reward': 10.0,
    'step_penalty': -0.01,
    'progress_reward': 0.5,
    'regress_penalty': -0.2,
    'invalid_action_penalty': -1.0,
    'timeout_penalty': -5.0,
}