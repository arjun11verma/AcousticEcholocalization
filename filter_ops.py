import numpy as np

def time_update_state(A, prev_state, B, control):
    return np.matmul(A, prev_state) + np.matmul(B, control)

def time_update_error(A, prev_P, Q):
    return np.matmul(np.matmul(A, prev_P), np.transpose(A)) + Q

def measurement_update_gain(prev_P, H, R):
    return np.matmul(np.matmul(prev_P, H), np.linalg.inv(np.matmul(np.matmul(H, prev_P), np.transpose(H)) + R))

def measurement_update_state(prev_state, K, z, H):
    return prev_state + np.matmul(K, z - np.matmul(H, prev_state))

def measurement_update_error(K, H, prev_P):
    return np.matmul(prev_P, np.identity(K.shape) - np.matmul(K, H))