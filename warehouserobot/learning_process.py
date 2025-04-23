import numpy as np
import matplotlib.pyplot as plt


import seaborn as sns

def plot_q_heatmap(q_values):
    max_q = np.max(q_values, axis=2)
    plt.figure(figsize=(8, 6))
    sns.heatmap(max_q, cmap="YlGnBu", annot=True, fmt=".1f")
    plt.title("Heatmap of Max Q-values")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.show()



