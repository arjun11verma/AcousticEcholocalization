from turtle import position
from matplotlib import pyplot as plt
import numpy as np

base_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

def compare_positions(position_vectors, covariances, labels):
    y_base = np.zeros(position_vectors[0].shape)
    y_offset = 0.05
    plt.axis([None, None, 0, y_offset + y_offset * len(position_vectors)])
    for i in range(len(position_vectors)):
        y_base.fill(y_offset * (i + 1))
        interval = np.sqrt(covariances[i])
        print(position_vectors[i] - interval)

        plt.plot([position_vectors[i] - interval, position_vectors[i] + interval], [y_base, y_base])
        plt.plot([position_vectors[i] - interval, position_vectors[i] - interval], [y_base + 0.0025, y_base - 0.0025])
        plt.plot([position_vectors[i] + interval, position_vectors[i] + interval], [y_base + 0.0025, y_base - 0.0025])
        plt.plot(position_vectors[i], y_base, 'o', color='#f44336')
    plt.show()