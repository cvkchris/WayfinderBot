import numpy as np
import os
import matplotlib.pyplot as plt

# Environment setup
rows, cols = 11, 11
actions = ['up', 'right', 'down', 'left']

rewards = np.full((rows, cols), -100.)
rewards[0, 5] = 100.  # Goal

aisles = {
    1: list(range(1, 10)),
    2: [1, 7, 9],
    3: list(range(1, 8)) + [9],
    4: [3, 7],
    5: list(range(cols)),
    6: [5],
    7: list(range(1, 10)),
    8: [3, 7],
    9: list(range(cols))
}

for r in aisles:
    for c in aisles[r]:
        rewards[r, c] = -1.

q_values = np.zeros((rows, cols, len(actions)))

# Hyperparameters
epsilon = 0.9
discount = 0.9
learning_rate = 0.9
episodes = 1000

# Track total rewards for each episode
total_rewards_per_episode = []

# Helper functions
def is_terminal_state(r, c):
    return rewards[r, c] != -1.

def get_start_location():
    while True:
        r = np.random.randint(rows)
        c = np.random.randint(cols)
        if not is_terminal_state(r, c):
            return r, c

def get_next_action(r, c, epsilon):
    if np.random.random() < epsilon:
        return np.argmax(q_values[r, c])
    return np.random.randint(len(actions))

def get_next_location(r, c, action):
    if actions[action] == 'up' and r > 0:
        r -= 1
    elif actions[action] == 'right' and c < cols - 1:
        c += 1
    elif actions[action] == 'down' and r < rows - 1:
        r += 1
    elif actions[action] == 'left' and c > 0:
        c -= 1
    return r, c

import seaborn as sns

def plot_q_heatmap(q_values):
    max_q = np.max(q_values, axis=2)
    plt.figure(figsize=(8, 6))
    sns.heatmap(max_q, cmap="YlGnBu", annot=True, fmt=".1f")
    plt.title("Heatmap of Max Q-values")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.show()


# Q-learning loop
for episode in range(episodes):
    r, c = get_start_location()
    episode_reward = 0  # Track reward for this episode
    
    while not is_terminal_state(r, c):
        action = get_next_action(r, c, epsilon)
        old_r, old_c = r, c
        r, c = get_next_location(r, c, action)
        reward = rewards[r, c]
        episode_reward += reward  # Accumulate reward for the episode
        old_q = q_values[old_r, old_c, action]
        best_future_q = np.max(q_values[r, c])
        q_values[old_r, old_c, action] = old_q + learning_rate * (reward + discount * best_future_q - old_q)

    total_rewards_per_episode.append(episode_reward)  # Save total reward for this episode

# Save Q-values to file
save_path = os.path.join(os.getcwd(), 'q_values.npy')
np.save(save_path, q_values)
print(f"✅ Q-table saved to: {save_path}")

# Print training configuration and reward structure
print("\n📋 Training Configuration")
print("-----------------------------")
print(f"🧠 Number of Episodes     : {episodes}")
print(f"📉 Learning Rate          : {learning_rate}")
print(f"🌀 Discount Factor (gamma): {discount}")
print(f"🎲 Exploration Rate (eps): {epsilon}")
print(f"📦 Q-table Shape          : {q_values.shape}")

# Print reward matrix summary
print("\n🏁 Reward System Overview")
print("-----------------------------")
print(f"🏆 Goal Reward     : {rewards[0, 5]}")
print(f"🚶‍♂️ Free Path Reward: -1")
print(f"🧱 Wall Penalty    : -100")

# Optional: count cells
num_walls = np.sum(rewards == -100)
num_paths = np.sum(rewards == -1)
num_goals = np.sum(rewards == 100)
print(f"\n📊 Environment Breakdown")
print("-----------------------------")
print(f"🧱 Wall Cells      : {num_walls}")
print(f"🚶‍♂️ Free Path Cells: {num_paths}")
print(f"🏁 Goal Cells      : {num_goals}")
