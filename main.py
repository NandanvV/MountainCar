import numpy as np
from agent import Agent
from grid_search import grid_search
from train import training
from config import q_learning_hyperparameters
from env import env, test_env

action_size = env.action_space.n
state_space = q_learning_hyperparameters["state_space"]

def testing_before_learning():
    state, _ = test_env.reset()
    done = False
    
    for _ in range(100):
        test_env.render()
        action = env.action_space.sample()
        state, reward, done, _, _ = test_env.step(action)
        if done:
            break

def main():
    while True:
        print("\nMountain Car Training Menu")
        print("1. Testing before learning (Random actions)")
        print("2. q-learning")
        print("3. SARSA")
        print("4. Grid Search for q-learning")
        print("5. Grid Search for SARSA")
        print("6. Quit")

        menu = input("Select: ")
        if menu == "1":
            testing_before_learning()       
        elif menu == "2":
            training(Agent(action_size, state_space, "q_learning"))
        elif menu == "3":
            training(Agent(action_size, state_space, "sarsa"))
        elif menu == "4":
            grid_search(action_size, "q_learning")
        elif menu == "5":
            grid_search(action_size, "sarsa")
        elif menu == "6":
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()