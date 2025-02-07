import numpy as np
from env import env

def training(agent, num_episodes=10000, rolling_window=100):
    episode_rewards = []

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        state = agent.discretize(state, env.observation_space.low, env.observation_space.high)
        done = False
        episode_reward = 0

        if agent.algorithm == "sarsa":
            action = agent.select_action(state)

        while not done:
            if agent.algorithm == "q_learning":
                action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = agent.discretize(next_state, env.observation_space.low, env.observation_space.high)

            if agent.algorithm == "sarsa":
                next_action = agent.select_action(next_state)
                agent.step(state, action, reward, next_state, next_action, done)
                action = next_action
            else:
                agent.step(state, action, reward, next_state, None, done)

            episode_reward += reward
            state = next_state

        episode_rewards.append(episode_reward)

        if len(episode_rewards) >= rolling_window:
            avg_reward = np.mean(episode_rewards[-rolling_window:])
            print(f"\rEpisode {episode}/{num_episodes} || Rolling Avg Reward: {avg_reward:.2f}", end="")

    print()

    return np.mean(episode_rewards[-rolling_window:])
