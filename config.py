algorithm_to_use = "q_learning"

q_learning_hyperparameters = {
    "alpha": 0.2,
    "gamma": 0.9,
    "eps_decay": 0.995,
    "state_space": [20, 20]
}

sarsa_hyperparameters = {
    "alpha": 0.05,
    "gamma": 0.95,
    "eps_decay": 0.999,
    "state_space": [20, 20]
}

def get_hyperparameters():
    return q_learning_hyperparameters if algorithm_to_use == "q_learning" else sarsa_hyperparameters

def update_algorithm(algorithm):
    global algorithm_to_use
    algorithm_to_use = algorithm