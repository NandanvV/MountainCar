import numpy as np
from config import q_learning_hyperparameters, sarsa_hyperparameters

class Agent:
    def __init__(self, action_size, state_space, algorithm="q_learning"):
        self.action_size = action_size
        self.state_space = state_space
        self.algorithm = algorithm

        if self.algorithm == "q_learning":
            params = q_learning_hyperparameters
        elif self.algorithm == "sarsa":
            params = sarsa_hyperparameters

        self.alpha = params["alpha"]
        self.gamma = params["gamma"]
        self.eps_decay = params["eps_decay"]

        self.eps = 1.0
        self.eps_min = 0.1
        self.q_table = np.zeros(state_space + [action_size])
        
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
    
