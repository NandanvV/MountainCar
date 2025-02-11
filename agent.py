import numpy as np
from config import algorithm_to_use, get_hyperparameters

class Agent:
    def __init__(self, action_size, Q_table, state_space=None, alpha=None, gamma=None, eps_decay=None):
        self.action_size = action_size
        self.state_space = state_space
        self.algorithm = algorithm_to_use

        params = get_hyperparameters()

        self.alpha = alpha if alpha is not None else params["alpha"]
        self.gamma = gamma if gamma is not None else params["gamma"]
        self.eps_decay = eps_decay if eps_decay is not None else params["eps_decay"]
        self.state_space = state_space if state_space is not None else params["state_space"]

        self.eps = 1
        self.eps_min = 0.1
        self.q_table = Q_table
        
    def discretize(self, state, env_low, env_high):
        bins = [np.linspace(env_low[i], env_high[i], self.state_space[i]) for i in range(len(state))]
        return tuple(np.digitize(state[i], bins[i]) - 1 for i in range(len(state)))
    
    def select_action(self, state):
        if np.random.rand() > self.eps:
            return np.argmax(self.q_table[state])
        else:
            return np.random.choice(range(self.action_size))
        
    def step(self, state, action, reward, next_state, next_action=None, done=False):

        former_q = self.q_table[state][action]

        if self.algorithm == "q_learning":
            next_q = np.max(self.q_table[next_state])

        elif self.algorithm == "sarsa":
            next_q = self.q_table[next_state][next_action] if next_action is not None else 0
            
        self.q_table[state][action] += self.alpha * (reward + self.gamma * next_q - former_q)

        if done:
            self.eps = max(self.eps_min, self.eps_decay * self.eps)
    
