# %%

# Import Statements

from turtle import forward
import visualization
import filter_ops
import numpy as np
from position_data import processing

# %%

# Data Reading and Processing

# constants
INITIAL_VELOCITY = 0.025
NUM_SAMPLES = 20
START_POS = 0.1

# data generation
ground_truth_positions = np.array([START_POS + i * INITIAL_VELOCITY for i in range(NUM_SAMPLES)])

real_impulse_responses = []
data_sample, hold = None, None
forward = True

for i in range(int(START_POS / INITIAL_VELOCITY), NUM_SAMPLES + int(START_POS / INITIAL_VELOCITY)):
    data_sample = processing.process_sample(i)
    if (abs(data_sample[1] - data_sample[0]) < 0.06):
        forward = False
    if (forward == False):
        hold = data_sample[0]
        data_sample[0] = data_sample[1]
        data_sample[1] = hold
    real_impulse_responses.append(data_sample)
real_impulse_responses = np.array(real_impulse_responses)

measurement_noise_covariance = processing.extract_noise_covariance(real_impulse_responses, ground_truth_positions)

# %%

# Kalman filter initialization

initial_state = np.array([0, 0]) # robot position, position of second end of pipe

Q = np.diag(np.array([0.15, 0.00001])) # Q - process noise covariance

control_matrix = np.array([1, 0]) # delta t

control_vector = 0.025 # control (velocity of robot)

A = np.identity(control_matrix.shape[0])

H = np.array([[1, 0], [-1, 1], [0, 1]]) # depends on direction and implementation!

estimation_uncertainty = np.diag(np.array([0, 100])) # initial uncertainty in filter

kalman_filter = filter_ops.KalmanFilter(measurement_noise_covariance, Q, A, control_matrix, H, control_vector, initial_state, estimation_uncertainty)

# %%

# Kalman filter prediction

kalman_positions = []
for i in range(NUM_SAMPLES):
    kalman_filter.kalman_update(real_impulse_responses[i])
    print(kalman_filter.get_state_vector())
    kalman_positions.append(kalman_filter.get_state_vector()[0])

labels = ['Ground truth positions', 'Noisy Measurements', 'Kalmann predicted positions']
position_vectors = [ground_truth_positions, real_impulse_responses[:, 0], np.array(kalman_positions)]

# %%

# Visualization

visualization.compare_positions(position_vectors, [0, 0.01, 0.01], labels)



# %%
