"""
Complex correlations
Authors: Arinc Demirel,Metehan Cekic
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import interactive
import math
# from tqdm import stqdm
import pdb
from tqdm import tqdm
import sys
from scipy import signal

from wifi_preamble import ideal_preamble_generation
from utils import *


def get_crosscorr(clustered_signals, signal_len, step_size, num_aligned_signal,ind_reference_transmitter, ind_reference_signal ):
    print("\n The length of the preamble is calculating: \n ")

    complex_signal_reference = clustered_signals[ind_reference_transmitter,
                                                 ind_reference_signal, :]
    complex_signals_random = np.zeros([num_aligned_signal, signal_len]) + 0j
    complex_signals_random_vector = np.zeros([num_aligned_signal,len(complex_signal_reference)//step_size,len(complex_signal_reference)]) + 0j
    number_of_experiment = signal_len//step_size

    index_of_random_complex_signal = []
    index_of_transmitter = []

    amplitude_peak = np.zeros([num_aligned_signal, number_of_experiment])

    correlations_with_zeros = np.zeros([number_of_experiment, 2 * signal_len - 1]) + 0j

    # index_of_random_complex_signal = np.zeros([1,num_aligned_signal])
    # index_of_transmitter = np.zeros([1,num_aligned_signal])

    iter_signals = tqdm(
        iterable=range(num_aligned_signal),
        unit="signal",
        leave=False)
    for k in iter_signals:

        found_new_signal = False
        while not found_new_signal:

            index_of_random_complex_signal.append(np.random.choice(
                clustered_signals.shape[1], size=1, replace=False))

            index_of_transmitter.append(np.random.choice(
                clustered_signals.shape[0], size=1, replace=False))

            if ind_reference_signal is not index_of_random_complex_signal[k] and ind_reference_transmitter is not index_of_transmitter[k]:

                found_new_signal = True

                # second complex signal from different transmitter
                complex_signals_random[k] = clustered_signals[index_of_transmitter[k],index_of_random_complex_signal[k], :]

                iter_pruning = tqdm(
                    iterable=range(0, signal_len, step_size),
                    unit="pruning",
                    leave=False)
                for i in iter_pruning:
                    complex_signals_random[k, signal_len-i:] = 0+0j

                    complex_signals_random_vector[:,i//step_size,:] = complex_signals_random[k]
                    
                    correlations_with_zeros[k, :] = signal.correlate(
                        complex_signal_reference, complex_signals_random[k], 'full')
                    
                    amplitude_peak[k, i//step_size] = np.max(np.abs(correlations_with_zeros))

    return amplitude_peak,complex_signals_random_vector


def main(step_size,num_aligned_signal, ind_reference_transmitter, ind_reference_signal, delta,type):

    signals, labels,fc_train,fc_test = load_data_from_npz(file_name="data.npz",type = type)
    comp_signals = signals[:, :, 0] + 1j*signals[:, :, 1]
    comp_signals = complex_normalize(comp_signals, norm= "l2")

    signals[:, :, 0] = comp_signals.real
    signals[:, :, 1] = comp_signals.imag
    # import pdb
    # pdb.set_trace()

    clustered_signals = cluster_signals(signals=signals, labels=labels)  # complex valued

    # ideal_preamble = ideal_preamble_generation(oversampling_rate=10, plot=False)

    # Data Information
    # num_signals = signals.shape[0]
    signal_len = signals.shape[1]
    amplitude_peak,complex_signals_random_vector = get_crosscorr(clustered_signals, signal_len, step_size,num_aligned_signal,ind_reference_transmitter,ind_reference_signal)

    plt.close('all')
    interactive(True)
    # np.random.seed(seed=0)
    np.set_printoptions(threshold = np.inf)
    
    amplitude_peak_mean = np.mean(amplitude_peak, axis=0)
    if channel_status == "channel"
        complex_signals_random_mean = np.mean(complex_signals_random_vector, axis=0)
        amplitude_peak_mean = mov_avg(amplitude_peak_mean, delta)

    
    for i in range(0,complex_signals_random_vector.shape[0]):
        complex_signals_random_mean[i,:] = np.mean(complex_signals_random_vector[i,:,:], axis=0)
   
    begining_point, length_from_tresholding, starting_point_from_tsh , ending_point_from_tresholding, length_from_binary_search, ending_point_from_binary_search,starting_point_from_bs = lengthEstimator(amplitude_peak_mean)


    interval1 = np.arange(0, signal_len, step_size)
    plt.figure()
    plt.plot(interval1, amplitude_peak_mean)
    plt.title('Normalized average peak points')

    plt.show()
    print(' The length of the preamble is  = {}'.format(length_from_binary_search))

    return begining_point, length_from_tresholding, starting_point_from_tsh , ending_point_from_tresholding, length_from_binary_search, ending_point_from_binary_search,starting_point_from_bs

if __name__ == '__main__':
    main(step_size = 10, num_aligned_signal = 3, ind_reference_transmitter = 18, ind_reference_signal = 199, delta = 40)
