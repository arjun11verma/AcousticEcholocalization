import numpy as np

# most basic iteration is a data vector that contains position, position of first echo source and position of second echo source

class KalmanFilter:
    def __init__(self, R, Q, A, B, H, control, state_vector, process_noise):
        self.R = R # measurement noise is predefined and constant
        self.Q = Q # process noise is predefined and constant
        self.A = A # relationship between state and next state
        self.B = B # relationship between control and state
        self.H = H # relationship between measurement and state 
        self.control = control
        self.state_vector = state_vector
        self.process_noise = process_noise
        self.kalman_gain = np.zeros(self.process_noise.shape) # determines weight of measurement vs process

    def time_update_state(self):
        self.state_vector = np.matmul(self.A, self.state_vector) + np.matmul(self.B, self.control)

    def time_update_error(self):
        self.process_noise = np.matmul(np.matmul(self.A, self.process_noise), np.transpose(self.A)) + self.Q

    def measurement_update_kalman_gain(self):
        self.kalman_gain = np.matmul(np.matmul(self.process_noise, self.H), np.linalg.inv(np.matmul(np.matmul(self.H, self.process_noise), np.transpose(self.H)) + self.R))

    def measurement_update_state(self, measurement):
        self.state_vector = self.state_vector + np.matmul(self.kalman_gain, measurement - np.matmul(self.H, self.state_vector))

    def measurement_update_error(self):
        self.process_noise = np.matmul(self.process_noise, np.identity(self.kalman_gain.shape) - np.matmul(self.kalman_gain, self.H))
    
    def get_state_vector(self):
        return self.state_vector