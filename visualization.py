from cProfile import label
from sys import intern
from turtle import position
from matplotlib import pyplot as plt
import numpy as np

base_colors = ['blue', 'green', 'red', 'yellow', 'purple']

def compare_positions(position_vectors, covariances, labels): 
    plt.clf()
    x_base = np.linspace(0, len(position_vectors[0]), len(position_vectors[0]))
    for i in range(len(position_vectors)):
        interval = np.sqrt(covariances[i])
        if (interval > 0):
            plt.plot([x_base, x_base], [position_vectors[i] - interval, position_vectors[i] + interval], color=base_colors[i])
            plt.plot([x_base + 0.25, x_base - 0.25], [position_vectors[i] - interval, position_vectors[i] - interval], color=base_colors[i])
            plt.plot([x_base + 0.25, x_base - 0.25], [position_vectors[i] + interval, position_vectors[i] + interval], color=base_colors[i])
        plt.plot(x_base, position_vectors[i], 'o', color=base_colors[i], label=labels[i])
    plt.legend()
    plt.show()

def plot_data(data, title):
    plt.clf()
    plt.plot(data)
    plt.title(title)
    plt.show()