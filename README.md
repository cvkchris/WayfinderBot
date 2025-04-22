# 🤖 WayfinderBot - Warehouse Navigation using Reinforcement Learning

WayfinderBot is a grid-based warehouse navigation simulation using Q-Learning. The bot learns to navigate efficiently from start to goal while avoiding obstacles (walls). It also visualizes the learned Q-values using heatmaps and tracks agent performance.

---

## 🚀 Project Overview

This project implements:
- A **grid world environment** simulating warehouse navigation
- **Q-learning** algorithm for training the agent
- Reward and penalty system for optimal path finding
- **Seaborn heatmaps** to visualize learned Q-values
- Easy-to-understand simulation and configuration

---

## 🧠 Core Features

| Feature                  | Description |
|--------------------------|-------------|
| `learning_process.py`    | Handles the Q-learning logic and training |
| `main.py`                | Visualizes the navigation and agent decisions |
| `q_values.npy`           | Stores the learned Q-table |
| 📊 Heatmaps              | Generated to understand learned behavior |
| 🧭 Goal, Wall, and Path  | Visual cues in environment |
| 💾 Configurable          | Change learning rate, gamma, epsilon easily |

---

## 🗂️ Project Structure
```
WayfinderBot/
├── warehouserobot/
│   ├── learning_process.py      # Q-learning training logic
│   ├── main.py                  # Environment setup and simulation
│   ├── q_values.npy             # Saved Q-table (after training)
│   └── __init__.py              # (optional) makes it a Python package
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## 🏗️ How to Run

1. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

2. **Train the bot**
    ```bash
    python warehouserobot/learning_process.py
    ```

3. **Run the main simulation**
    ```bash
    python warehouserobot/main.py
    ```

---

## 📉 Training Configuration (Default)

| Parameter         | Value  |
|------------------|--------|
| Episodes          | 1000   |
| Learning Rate     | 0.9    |
| Discount Factor γ | 0.9    |
| Exploration Rate ε| 0.9    |

---

## 🏆 Reward System

| Scenario       | Reward   |
|----------------|----------|
| Reaching Goal  | +100     |
| Moving Freely  | -1       |
| Hitting a Wall | -100     |

---

