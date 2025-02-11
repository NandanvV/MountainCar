from collections import defaultdict
import itertools
import json

import numpy as np
from agent import Agent
from train import training
from config import algorithm_to_use

def grid_search(action_size, num_episodes=7500, rolling_window=100):
    
    alpha_values = [0.01, 0.05, 0.1, 0.2]
    gamma_values = [0.9, 0.95, 0.99, 0.999]
    eps_decay_values = [0.99, 0.995, 0.999]
    state_space_values = [[10, 10], [20, 20], [30, 30]]

    hyperparameter_combinations = list(itertools.product(alpha_values, gamma_values, eps_decay_values, state_space_values))

    results = []

    print("\nRunning Grid Search...")
    for idx, (alpha, gamma, eps_decay, state_space) in enumerate(hyperparameter_combinations, start=1):
        print(f"Iteration {idx} / {len(hyperparameter_combinations)} - Testing: Alpha={alpha}, Gamma={gamma}, Eps Decay={eps_decay}, State Space={state_space}")
        Q_table = defaultdict(lambda: np.zeros(action_size))
        agent = Agent(action_size, Q_table, state_space=state_space, alpha=alpha, gamma=gamma, eps_decay=eps_decay)
        avg_reward = training(agent, num_episodes=num_episodes, rolling_window=rolling_window)
        results.append((alpha, gamma, eps_decay, state_space, avg_reward))

    # Find the best hyperparameter combination
    best_params = max(results, key=lambda x: x[4])
    print(f"\nBest Parameters Found: Alpha={best_params[0]}, Gamma={best_params[1]}, "
          f"Eps Decay={best_params[2]}, State Space={best_params[3]}, Avg Reward={best_params[4]:.2f}")
    
     # Save best hyperparameters to `best_hyperparams.json`
    best_hyperparams = {
        "algorithm": algorithm_to_use,
        "alpha": best_params[0],
        "gamma": best_params[1],
        "eps_decay": best_params[2],
        "state_space": best_params[3],
        "avg_reward": best_params[4]
    }

    with open("best_hyperparams.json", "w") as f:
        json.dump(best_hyperparams, f, indent=4)

    print("\nBest hyperparameters saved to `best_hyperparams.json`")
