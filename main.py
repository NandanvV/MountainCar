from collections import defaultdict
import numpy as np
from agent import Agent
from grid_search import grid_search
from train import training
from config import update_algorithm
from env import env, test_env
from test import test_agent

action_size = env.action_space.n

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
    Q_table = defaultdict(lambda: np.zeros(env.action_space.n))

    while True:
        print("\nMountain Car Training Menu")
        print("1. Testing before learning (Random actions)")
        print("2. Train using Q-learning")
        print("3. Train using SARSA")
        print("4. Grid Search for Q-learning")
        print("5. Grid Search for SARSA")
        print("6. Test trained agent")
        print("7. Quit")

        menu = input("Select: ")
        if menu == "1":
            testing_before_learning()       
        elif menu == "2":
            update_algorithm("q_learning")
            Q_table.clear()
            agent = Agent(action_size, Q_table)
            training(agent)
        elif menu == "3":
            update_algorithm("sarsa")
            Q_table.clear()
            agent = Agent(action_size, Q_table)
            training(agent)
        elif menu == "4":
            update_algorithm("q_learning")
            grid_search(action_size)
        elif menu == "5":
            update_algorithm("sarsa")
            grid_search(action_size)
        elif menu== "6":
            test_agent(Agent(action_size, Q_table))
        elif menu == "7":
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == "__main__":
    main()