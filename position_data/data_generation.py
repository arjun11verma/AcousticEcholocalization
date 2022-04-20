import numpy as np

def generate_ground_truth_random(num_positions, velocity, pipe_length): # generating ground truth positions
    positions = np.empty(num_positions)
    positions[0] = 0

    for i in range(1, num_positions):
        next_position = positions[i - 1] + (np.random.random()) * velocity
        positions[i] = next_position if next_position < pipe_length else pipe_length
    
    return positions

def generate_impulse_responses(gt_positions, pipe_length):
    impulse_responses = np.empty((gt_positions.shape[0], 3))

    for i in range(0, impulse_responses.shape[0]):
        impulse_responses[i][0] = gt_positions[i]
        impulse_responses[i][1] = pipe_length - gt_positions[i]
        impulse_responses[i][2] = pipe_length
    
    return impulse_responses # position of robot, distance from robot to end of pipe, length of pipe

def add_noise_impulse_responses(impulse_responses, noise_covariance):
    noisy_impulse_responses = np.empty(impulse_responses.shape)
    for i in range(impulse_responses.shape[1]):
        noisy_response = impulse_responses[:, i] + np.random.normal(0, np.sqrt(noise_covariance[i][i]), impulse_responses[:, i].shape[0])
        noisy_impulse_responses[:, i] = np.clip(noisy_response, 0, None)
    return noisy_impulse_responses

    
