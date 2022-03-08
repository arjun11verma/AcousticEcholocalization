import numpy as np

def generate_ground_truth_random(num_measurements, velocity): # generating ground truth positions
    measurements = np.empty((num_measurements, 1))
    measurements[0][0] = 0

    for i in range(1, num_measurements):
        measurements[i][0] = measurements[i - 1][0] + (np.random.random()) * velocity
    
    return measurements

def generate_ground_truth_constant_velocity(num_measurements, velocity): # generating ground truth positions
    measurements = np.empty((num_measurements, 1))
    measurements[0][0] = 0

    for i in range(1, num_measurements): # assuming delta t is one second
        measurements[i][0] = measurements[i - 1][0] + velocity
    
    return measurements

def generate_impulse_responses(gt_positions, pipe_length):
    impulse_responses = np.empty((gt_positions.shape[0], 3))

    for i in range(0, impulse_responses.shape[0]):
        impulse_responses[i][0] = gt_positions[i][0]
        impulse_responses[i][1] = pipe_length - gt_positions[i][0]
        impulse_responses[i][2] = pipe_length
    
    return impulse_responses

def add_noise_impulse_responses(impulse_responses, noise_covariance):
    noisy_impulse_responses = np.empty(impulse_responses.shape)
    for i in range(impulse_responses.shape[1]):
        noisy_response = impulse_responses[:, i] + np.random.normal(0, np.sqrt(noise_covariance[i][i]), impulse_responses[:, i].shape[0])
        noisy_impulse_responses[:, i] = noisy_response
    return noisy_impulse_responses

    
