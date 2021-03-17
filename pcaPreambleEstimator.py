"""
Complex correlations
Authors: Arinc Demirel
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import PCA
from matplotlib import interactive
import math
from scipy import signal


from wifi_preamble import ideal_preamble_generation
import lengthFinder as lengthFinder
from lengthFinder import *
from AddChannel import *
from utils import *

    
class PcA():

    def __init__(self, complex_signals_clustered, num_aligned_signal, ind_reference_transmitter,ind_reference_signal ):

        self.num_aligned_signal = num_aligned_signal
        self.ind_reference_transmitter = ind_reference_transmitter  # Reference transmitter index
        self.ind_reference_signal = ind_reference_signal  # Reference signal index
        self.complex_signal_reference = complex_signals_clustered[self.ind_reference_transmitter, self.ind_reference_signal, :]
        self.signal_len = len(self.complex_signal_reference)

    def get_cos_sim(self,complex_signals_clustered):
        """
        Looking cosine similarity between the signals with selecting one of them reference and looking for the CS between reference and  other signals from other transmitters 
        the first values of peaks are negligable thus the peaks are looking from 100 to 1100 
        """
        peak_points = []
        peak_points2 = []
        peak_points3 = []
        ind_aligned_signals = []

        index_of_random_complex_signal = []
        index_of_transmitter = []

       
        signal_correlations = np.zeros([self.num_aligned_signal, 2*(self.signal_len)-1])+ 0j
        signal_correlations2 = np.zeros([self.num_aligned_signal, 2*(self.signal_len)-1])+ 0j
        complex_signals_random = np.zeros([self.num_aligned_signal, self.signal_len]) + 0j
        normalized_correlations = np.zeros([self.num_aligned_signal, 2*(self.signal_len)-1])+ 0j
        normalized_correlations_abs = np.zeros([self.num_aligned_signal, 2*(self.signal_len)-1])+ 0j
        j = 0

        while j <= self.num_aligned_signal  - 1:

            index_of_random_complex_signal.append(np.random.choice(complex_signals_clustered.shape[1], size=1, replace=False))
            index_of_transmitter.append(np.random.choice(complex_signals_clustered.shape[0], size=1, replace=False))

            if index_of_random_complex_signal[j] is not self.ind_reference_signal and index_of_transmitter[j] is not self.ind_reference_transmitter:

                complex_signals_random[j] = complex_signals_clustered[index_of_transmitter[j], index_of_random_complex_signal[j], : ]    # second complex signal from different transmitter

                signal_correlations[j] = np.correlate(np.abs(complex_signals_random[j]),np.abs(self.complex_signal_reference), 'full')
                signal_correlations2[j] = signal.correlate(complex_signals_random[j],self.complex_signal_reference,'full')
                # plt.figure()
                # plt.plot(np.abs(signal_correlations[j]))
                # plt.title('Signal Correlations j = {}'.format(j))
                peak_points.append(np.argmax(signal_correlations[j]))

                normalized_correlations[j] = NCC(self.complex_signal_reference, complex_signals_random[j])
                normalized_correlations_abs[j] = np.abs(normalized_correlations[j])
                peak_points2.append( np.argmax(normalized_correlations_abs[j,round(len(normalized_correlations_abs[j])/4): round(len(normalized_correlations_abs[j])/6)*5]) + round(len(normalized_correlations_abs[j])/4) )
                peak_points3.append(np.argmax(np.abs(signal_correlations2[j])))
                # plt.figure()
                # plt.plot(np.abs(normalized_correlations[j]))
                # plt.title('Normalized Correlations j = {}'.format(j))
                # plt.figure()
                # plt.plot(np.abs(self.complex_signal_reference))
                # plt.title('Reference Signal')
                # plt.figure()
                # plt.plot(np.abs(complex_signals_random[j]))
                # plt.title('Complex signals random j = {}'.format(j))
                # breakpoint()
                j += 1

        print('Index of peak points from total signal', peak_points)
        # print('Index of peak points from total signal', peak_points2, np.mean(peak_points2))
        # print('Index of peak points from total signal', peak_points3, np.mean(peak_points3))

        
        maxi = np.max(peak_points)
        differ = maxi - peak_points

        max_len = len(complex_signals_random[1]) + np.max(differ)
        # max_len = round(np.mean(peak_points2))


        return maxi,differ,max_len,complex_signals_random,peak_points,peak_points2


    def aligner(self, differ, complex_signals_random):

        corr2 = np.zeros([self.num_aligned_signal, (2*len(complex_signals_random[1]) + np.max(differ))-1])
        extended_complex_signals = np.zeros([self.num_aligned_signal,len(complex_signals_random[1]) + np.max(differ)]) + 0j
        peak_changed = []

        for i in (range(len(differ))):

            extended_complex_signals[i][differ[i]:differ[i] + len(complex_signals_random[i])] = complex_signals_random[i]
            
            # corr2[i] = np.correlate(extended_complex_signals[i], self.complex_signal_reference, 'full')
            # peak_changed = np.argmax(np.abs(corr2[i]))

            # print('The location of the peak point was at {} and is changed to the {}'.format(peak_points[i], peak_changed))

        return extended_complex_signals


def PCA_estimator(extended_complex_signals, length, n_components = 3):


    pca = PCA(n_components= n_components)
    
    principalComponent_real = pca.fit_transform(np.real(extended_complex_signals[0:length]).T)
    pca_val_real = pca.singular_values_
    principalComponent_real = principalComponent_real.T

    principalComponent_imag = pca.fit_transform(np.imag(extended_complex_signals[0:length]).T)
    pca_val_imag = pca.singular_values_
    principalComponent_imag = principalComponent_imag.T

    principalComponent = principalComponent_real + 1j*principalComponent_imag


    return principalComponent,pca_val_imag

	
def main(step_size,num_aligned_signal, ind_reference_transmitter, ind_reference_signal,type):

    signals, labels,fc_train,fc_test = load_data_from_npz(file_name="data.npz",type = type )
    comp_signals = signals[:, :, 0] + 1j*signals[:, :, 1]
    comp_signals = complex_normalize(comp_signals)
    signals[:, :, 0] = comp_signals.real
    signals[:, :, 1] = comp_signals.imag
    # import pdb
    # pdb.set_trace()
    clustered_signals = cluster_signals(signals=signals, labels=labels)  # complex valued
    # ideal_preamble = ideal_preamble_generation(oversampling_rate=10, plot=False)
   

    PCA_estimation = PcA(clustered_signals, num_aligned_signal= num_aligned_signal, ind_reference_transmitter = ind_reference_transmitter, ind_reference_signal = ind_reference_signal )
    maxi,differ,max_len,complex_signals_random,peak_points,peak_points2 = PCA_estimation.get_cos_sim(clustered_signals)
    aligned_signals = PCA_estimation.aligner(differ, complex_signals_random)

    
    begining_point, length_from_tresholding, starting_point_from_tsh , ending_point_from_tresholding,length_from_binary_search, ending_point_from_binary_search,starting_point_from_bs = lengthFinder.main(step_size , num_aligned_signal , ind_reference_transmitter, ind_reference_signal , delta = 40,type = type)

    estimated_signals_from_pca,pca_val = PCA_estimator(aligned_signals,length_from_binary_search, n_components = 3)

    interactive(True)
    # np.random.seed(seed=0)
    np.set_printoptions(threshold=np.inf)

    # original_preamble = ideal_preamble_generation(oversampling_rate=10, plot=False)
    # original_preamble_real = np.real(original_preamble)

    # test_samples_real = np.real(complex_signal_reference)
    # test_samples_real_preamble = test_samples_real.copy()
    # test_samples_real_preamble[3200:] = 0
    estimated_preamble = estimated_signals_from_pca[0, starting_point_from_bs:length_from_binary_search+starting_point_from_bs]
    plt.figure()
    plt.plot(complex_normalize(estimated_preamble, norm="l2"), 'b')  # 256 best view
    plt.plot(complex_normalize(PCA_estimation.complex_signal_reference[ starting_point_from_bs:length_from_binary_search+starting_point_from_bs], norm="l2"), 'y')  # 78+33+40 best view
    plt.title('Reference Preamble Real(y) vs Principal Component Real(b)')
    plt.show()

    similar = cos_sim(np.abs(PCA_estimation.complex_signal_reference[starting_point_from_bs:length_from_binary_search+starting_point_from_bs]), np.abs(estimated_signals_from_pca[0, starting_point_from_bs:length_from_binary_search+starting_point_from_bs]))
    print("Cosine simalarity between the principal component pramble  and the reference signal preamble is {}".format(similar))

    return estimated_preamble, labels , clustered_signals,length_from_binary_search
    
    signal_preambles =  preambleDetectionFromDataset(clustered_signals, estimated_preamble, length_from_binary_search)
   
    mixed_preambles,new_one_hot_labels =  mixData(signal_preambles,labels)


    return 
"""
Experiments / Take them to another .py file 
"""
# experiments=[]
# lengt_experiments_from_tsh = []
# lengt_experiments_from_bs = []


# experiment_number = 1
# val_num_aligned_signals = int(input("\n Enter the number of signals: "))
# peakpoints_average = np.zeros([experiment_number,val_num_aligned_signals])
# peakpoints2_average = np.zeros([experiment_number,val_num_aligned_signals])


# for i in range(0,experiment_number):
#     signal_preambles, mixed_preambles, new_one_hot_labels = main(step_size = 10, num_aligned_signal = val_num_aligned_signals, ind_reference_transmitter=18, ind_reference_signal=199)


#     experiments.append(similar)
#     lengt_experiments_from_tsh.append(length_from_tresholding)
#     lengt_experiments_from_bs.append(length_from_binary_search)
#     peakpoints_average[i,:] = peak_points
#     peakpoints2_average[i,:] = peak_points2

# mean = np.mean(peakpoints_average, axis=1)
# mean2 = np.mean(peakpoints2_average, axis=1)
# print(' The mean of cosine similarities ', sum(experiments)/len(experiments))
# print(' The mean of lengths calculated with the help of binary search ',sum(lengt_experiments_from_bs)/len(lengt_experiments_from_bs))
# print(' The mean of lengths calculated with the help of tresholding ',sum(lengt_experiments_from_tsh)/len(lengt_experiments_from_tsh))

# standard_deviation_of_bs = math.sqrt(sum((lengt_experiments_from_bs-np.mean(lengt_experiments_from_bs))**2/(len(lengt_experiments_from_bs))))
# standard_deviation_of_cosine_similarity = math.sqrt(sum((experiments-np.mean(experiments))**2/(len(experiments))))
# plt.figure()
# plt.hist(lengt_experiments_from_bs)
# plt.title('Histogram of the lengths with algorithm with number of signals {}'. format(val_num_aligned_signals))
# plt.figure()
# plt.hist(experiments)
# plt.title('Histogram of the cosine similarities with number of signals {}'. format(val_num_aligned_signals))


