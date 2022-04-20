from email.mime import base
import matplotlib
import numpy as np
import csv
from scipy import signal as sig
from matplotlib import pyplot as plt


################ We just define some useful functions ################
# It is not necessary to understand what follows

def getDFTMatrix(M):
    """
    Compute the Discrete Fourier Transform matrix of size M
    """
    omega = np.exp(-2j*np.pi/M)
    W = np.asarray([m for m in range(M)])
    W.shape = (M, 1)
    W = W.dot(np.transpose(W))
    W = W[:int((M+1)/2),:]
            
    return 2*(omega**W)

def createEmissionSignal(N, f, Ncycles, Time, HannWindow=True, Fe=1.25e6):
    """
    Create an excitation signal with a sinusoïdal shape 
    - N : size of the signal 
    - f : Central frequency of the signal
    - Ncycles : Number of tonebursts of the signal
    - Time : 1D vector that contains the values of time 
    - HanmWindow : Booleen that is true to enclose the signal in an Hanning window
    """ 
    Signal = 0*Time
    Nsin = int( (Ncycles*Fe)//f )
    if HannWindow:
        Signal[:Nsin] = np.sin(np.pi*(np.arange(Nsin).transpose())/Nsin)**2 * np.sin(2*np.pi*f*Time[:Nsin]) 
    else:
        Signal[:Nsin] = np.sin(2*np.pi*f*Time[:Nsin]) 
    Signal = Signal/max(abs(Signal))
       
    return Signal

def getPropagationFunction(LWA, LPoints):
    """
    """    
    return np.exp(-1j*LWA*LPoints) / np.sqrt(0.5+LPoints*LWA) 

def correct_data_length(data):
    if (data.shape[0] > 3): 
        data = data[:3]
        return data
    else:
        diff = abs(data[0] - data[1])
        new_data = [data[0], diff, data[1]] if diff > data[0] else [diff, data[0], data[1]]
        return new_data

    

################ Data processing #################

def process_sample(sample_number):
    correlation = lambda s1, s2 : np.sum(s1*s2, axis=0) / (np.linalg.norm(s1, axis=0)* np.linalg.norm(s2, axis=0))

    Ts = 8e-7 # Sampling period
    M = 2499 # Size of the data
    Time = np.arange(M)*Ts
    LWA = np.load("LWA.npy").reshape(-1, 1)

    W = getDFTMatrix(M) # DFT matrix
    iW = (1./(2*M))*W.transpose().conj() # inverse DFT Matrix

    excitation_signal = createEmissionSignal(M, 4.5e4, 5, Time, True).reshape(-1, 1) 
    Fexcitation = W.dot(excitation_signal)

    # Create a dictionnary of normalized signals using the acoustic model for specific propagation distances
    Dist = np.linspace(0, 1.2, 300) # List of propagation distances
    Dict = iW.dot( Fexcitation*getPropagationFunction(LWA, 2*Dist) ) # The distance are multiplied
    Dict0 = Dict.real
    Dict = Dict0 / np.sqrt(np.sum(Dict0**2, axis=0))


    # Example on how to extract ehoes from one the file
    data = np.genfromtxt(f'C:\Research\GTLorraine\echodata\{sample_number}.csv', delimiter=",")[:M, 1].reshape(-1, 1)
    data = data - np.mean(data); data[:180, :] = 0
    corr = np.abs(sig.hilbert(correlation(Dict, data)))

    prominence = 0.05 # This is a parameter you can modify probabiblt within the interval [0.01 0.2]
    peaks, properties = sig.find_peaks(corr, prominence=prominence)
    ranges_to_reflector = Dist[peaks] # This the output with ranges from a reflector
    ranges_to_reflector = ranges_to_reflector
    
    if (len(ranges_to_reflector) != 3): 
        ranges_to_reflector = correct_data_length(ranges_to_reflector)

    return ranges_to_reflector

def extract_noise_covariance(noisy_data, ground_truth, PIPE_LENGTH = 1):
    ground_truth_echos = np.zeros((ground_truth.shape[0], 3))
    ground_truth_echos[:, 0] = ground_truth
    ground_truth_echos[:, 2] = PIPE_LENGTH
    ground_truth_echos[:, 1] = ground_truth_echos[:, 2] - ground_truth_echos[:, 0]

    print(noisy_data, ground_truth_echos)

    noise_covariances = []
    for i in range(3):
        noise = noisy_data[:, i] - ground_truth_echos[:, i]
        noise_covariances.append(np.var(noise))
    noise_covariances = np.array(noise_covariances)

    return np.diag(noise_covariances)