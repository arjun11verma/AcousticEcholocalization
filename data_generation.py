import numpy as np

# most basic iteration is a data vector that contains position, position of first echo source and position of second echo source

def generate_true_measurements(num_measurements, initial_measurement, movement_range):
    measurements = np.empty((num_measurements, initial_measurement.size))
    measurements[0][:] = initial_measurement

    for i in range(1, num_measurements):
        measurements[i][1:] = measurements[i - 1][1:]
        measurements[i][1] = measurements[i - 1][1] + 2 * (0.5 - np.random.random()) * movement_range
    
    return measurements

def add_noise(noise_covariance, measurements):
    pass

def generate_noise_covariance():
    pass
    
    
