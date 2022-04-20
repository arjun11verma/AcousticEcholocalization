from enum import Enum
import numpy as np

class Direction(Enum):
    FORWARD = 0
    BACKWARD = 1

# YOUR H RELATION IS BACKWARDS! STATE TIMES H EQUALS MEASUREMENT!
# ADD ASSOCIATION TO MEASUREMENT

# RELATE STATE INTO MEASUREMENT BY H AND THEN CHECK, have a cutoff as well so you can put the measurement ins the correct order!

class KalmanFilter:
    def __init__(self, R, Q, A, B, H, control, state_vector, estimation_uncertainty):
        self.R = R # measurement noise is predefined and constant
        self.Q = Q # process noise - parameter of the filter!
        self.A = A # relationship between state and next state
        self.B = B # relationship between control and state
        self.H = H # relationship between state and measurement
        self.estimation_uncertainty = estimation_uncertainty # estimation uncertainty

        self.direction = Direction.FORWARD # current direction
        self.control = control
        self.state_vector = state_vector
        
        self.kalman_gain = np.zeros(self.estimation_uncertainty.shape) # determines weight of measurement vs process

    def time_update_state(self):
        self.state_vector = np.matmul(self.A, self.state_vector) + self.B * self.control

    def time_update_error(self):
        self.estimation_uncertainty = np.matmul(np.matmul(self.A, self.estimation_uncertainty), np.transpose(self.A)) + self.Q

    def measurement_update_kalman_gain(self):
        self.kalman_gain = np.matmul(np.matmul(self.estimation_uncertainty, np.transpose(self.H)), np.linalg.inv(np.matmul(np.matmul(self.H, self.estimation_uncertainty), np.transpose(self.H)) + self.R))

    def measurement_update_state(self, measurement):
        self.state_vector = self.state_vector + np.matmul(self.kalman_gain, measurement - np.matmul(self.H, self.state_vector))

    def measurement_update_error(self):
        self.estimation_uncertainty = np.matmul(self.estimation_uncertainty, np.identity(self.estimation_uncertainty.shape[0]) - np.matmul(self.kalman_gain, self.H))
    
    def get_state_vector(self):
        return self.state_vector

    def kalman_update(self, measurement):
        self.time_update_state()
        self.time_update_error()
        self.measurement_update_kalman_gain()
        self.measurement_update_state(measurement)
        self.measurement_update_error()