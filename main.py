import data_generation
import filter_ops
import numpy as np

# predefined constants
PIPE_LENGTH = 20
INITIAL_VELOCITY = 0.8
NUM_SAMPLES = 20
measurement_noise_covariance = np.diag(np.array([0.1, 0.1, 0.1])) # predefined for simulation, would be calculated in real life

# kalman filter
initial_state = np.array([0, 0, 0]) # position is at 0, first end of pipe is at 0, last end of pipe is at 0?
motion_uncertainty_covariance = np.diag(np.array([0.075, 0.075, 0.075])) # Q
control_matrix = np.diag(np.array([INITIAL_VELOCITY, 0, 0])) # B
control_vector = np.array([1, 1, 1]) # control
A = np.identity(control_matrix.shape[0]) 
H = np.diag(np.array([1, 0, 1]))
kalman_filter = filter_ops.KalmanFilter(measurement_noise_covariance, motion_uncertainty_covariance, A, control_matrix, H, control_vector, initial_state, np.zeros(control_matrix.shape))

# data generation
ground_truth_positions = data_generation.generate_ground_truth_random(NUM_SAMPLES, INITIAL_VELOCITY)
impulse_responses = data_generation.generate_impulse_responses(ground_truth_positions, PIPE_LENGTH)
noisy_impulse_responses = data_generation.add_noise_impulse_responses(impulse_responses, measurement_noise_covariance)

# kalman prediction
for i in range(NUM_SAMPLES):
    print(f'Kalmann predicted position: {kalman_filter.get_state_vector()[0]}, Actual position: {ground_truth_positions[i]}')
    kalman_filter.kalman_update(noisy_impulse_responses[i])



