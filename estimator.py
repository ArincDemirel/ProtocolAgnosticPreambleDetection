"""
Complex correlations
Authors: Arinc Demirel
Contributer: Metehan Cekic
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import interactive
import math
from tqdm import tqdm
import pdb
from tqdm import tqdm

from wifi_preamble import ideal_preamble_generation
from utils import *


def complex_normalize(signals, norm="l2"):
    """
    Normalizer function for complex signals
    """

    if norm == "l2":
        magnitudes = np.abs(signals)
        lp_norm = np.sqrt((magnitudes**2).sum(axis=-1))
    elif norm == "l1":
        magnitudes = np.abs(signals)
        lp_norm = magnitudes.sum(axis=-1)
    elif norm == "inf":
        lp_norm = np.max(np.abs(signals), axis=-1)
    else:
        raise NotImplementedError
    if len(signals.shape) > 1:
        lp_norm = lp_norm.reshape(lp_norm.shape+(1,))
    signals /= lp_norm
    return signals


def load_data_from_npz(file_name="data.npz"):
    """
    Load data from npz file
    """
    data = np.load(file_name)

    # arr0 - x train
    # arr1 - y train

    signals = data['arr_0']
    one_hot_labels = data['arr_1']  # in one hot form
    labels = np.argmax(one_hot_labels, axis=1)

    return signals, labels


def cluster_signals(signals, labels):
    """
    Cluster signals according to their labels
    """

    complex_form = signals[:, :, 0] + 1j*signals[:, :, 1]

    # Data Information
    num_signals = signals.shape[0]
    signal_len = signals.shape[1]
    num_transmitters = np.max(labels) + 1

    complex_signals_clustered = np.zeros(
        [num_transmitters, num_signals//num_transmitters, signal_len]) + 0j
    transmitter_signal_indices = np.zeros(num_transmitters, dtype=np.int32)

    for i in range(num_signals):
        complex_signals_clustered[labels[i],
                                  transmitter_signal_indices[labels[i]], :] = complex_form[i]
        transmitter_signal_indices[labels[i]] += 1

    return complex_signals_clustered


def get_crosscorr(clustered_signals, signal_len, step_size=10):

    num_aligned_signal = 19
    ind_reference_transmitter = 18  # Reference transmitter index
    ind_reference_signal = 199  # Reference signal index

    complex_signal_reference = clustered_signals[ind_reference_transmitter,
                                                 ind_reference_signal, :]
    complex_signals_random = np.zeros([num_aligned_signal, signal_len]) + 0j

    number_of_experiment = signal_len//step_size

    index_of_random_complex_signal = []
    # index_of_transmitter = []

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
            index_of_transmitter = ind_reference_transmitter
            # index_of_transmitter.append(np.random.choice(
            #     clustered_signals.shape[0], size=1, replace=False))

            # if ind_reference_signal is not index_of_random_complex_signal[k] and ind_reference_transmitter is not index_of_transmitter[k]:
            if ind_reference_signal is not index_of_random_complex_signal[k] :

                found_new_signal = True

                # second complex signal from different transmitter
                complex_signals_random[k] = clustered_signals[index_of_transmitter,
                                                              index_of_random_complex_signal[k], :]

                iter_pruning = tqdm(
                    iterable=range(0, signal_len, step_size),
                    unit="pruning",
                    leave=False)
                for i in iter_pruning:
                    complex_signals_random[k, signal_len-i:] = 0+0j

                    correlations_with_zeros[k, :] = np.correlate(
                        complex_signal_reference, complex_signals_random[k], 'full')

                    amplitude_peak[k, i//step_size] = np.max(np.abs(correlations_with_zeros))

    return amplitude_peak


def main():

    signals, labels = load_data_from_npz(file_name="data.npz")
    comp_signals = signals[:, :, 0] + 1j*signals[:, :, 1]
    comp_signals = complex_normalize(comp_signals)
    breakpoint()
    signals[:, :, 0] = comp_signals.real
    signals[:, :, 1] = comp_signals.imag
    # import pdb
    # pdb.set_trace()

    clustered_signals = cluster_signals(signals=signals, labels=labels)  # complex valued
    # ideal_preamble = ideal_preamble_generation(oversampling_rate=10, plot=False)

    # Data Information
    # num_signals = signals.shape[0]
    signal_len = signals.shape[1]
    # num_transmitters = np.max(labels) + 1

    step_size = 10
    amplitude_peak = get_crosscorr(clustered_signals, signal_len, step_size=step_size)

    plt.close('all')
    interactive(True)
    # np.random.seed(seed=0)
    np.set_printoptions(threshold=np.inf)

    amplitude_peak_mean = np.mean(amplitude_peak, axis=0)
    # amplitude_peak_real_mean = np.mean(amplitude_peak_real, axis=0)
    # amplitude_peak_imag_mean = np.mean(amplitude_peak_imag, axis=0)

    interval1 = np.arange(0, signal_len, step_size)

    plt.figure()
    plt.plot(interval1, amplitude_peak_mean)
    plt.title('Normalized average peak points')
    plt.savefig("corrs.pdf")


if __name__ == '__main__':
    main()
