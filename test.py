import numpy as np
from env import env 

def test_agent(agent, num_episodes=100, render=True):   
    agent.eps = 0 
    total_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = agent.discretize(state, env.observation_space.low, env.observation_space.high)
        done = False
        episode_reward = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = agent.discretize(next_state, env.observation_space.low, env.observation_space.high)

            episode_reward += reward
            state = next_state

        total_rewards.append(episode_reward)

    avg_reward = np.mean(total_rewards)
    print(f"\nAverage Reward over {num_episodes} episodes: {avg_reward:.2f}")
    return avg_reward