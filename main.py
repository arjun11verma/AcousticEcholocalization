import visualization
import data_generation
import filter_ops
import numpy as np

# predefined constants
PIPE_LENGTH = 20
INITIAL_VELOCITY = 0.8
NUM_SAMPLES = 20
measurement_noise_covariance = np.diag(np.array([0.1, 0.1, 0.1])) 
# predefined for simulation, would be calculated in real life

# kalman filter
initial_state = np.array([0, 0, 0]) # robot position, position of first end of pipe, position of second end of pipe
motion_uncertainty_covariance = np.diag(np.array([0.015, 0.00001, 0.00001])) # Q
control_matrix = np.array([1, 1, 1]) # delta t
control_vector = INITIAL_VELOCITY # control (velocity of robot)
A = np.identity(control_matrix.shape[0]) 
H = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]]) # depends on direction!
estimation_uncertainty = np.diag(np.array([100, 100, 100]))

kalman_filter = filter_ops.KalmanFilter(measurement_noise_covariance, motion_uncertainty_covariance, A, control_matrix, H, control_vector, initial_state, estimation_uncertainty)

# data generation
ground_truth_positions = data_generation.generate_ground_truth_random(NUM_SAMPLES, INITIAL_VELOCITY, PIPE_LENGTH)
impulse_responses = data_generation.generate_impulse_responses(ground_truth_positions, PIPE_LENGTH)
noisy_impulse_responses = data_generation.add_noise_impulse_responses(impulse_responses, measurement_noise_covariance)

# kalman prediction
kalman_positions = []
for i in range(NUM_SAMPLES):
    print(f'Kalmann predicted position: {kalman_filter.get_state_vector()[0]}, Actual position: {ground_truth_positions[i]}')
    kalman_positions.append(kalman_filter.get_state_vector()[0])
    kalman_filter.kalman_update(noisy_impulse_responses[i])

labels = ['Ground truth positions', 'Kalmann predicted positions']
position_vectors = [ground_truth_positions, np.array(kalman_positions)]

visualization.compare_positions(position_vectors, [0.1, 0.1], labels)


