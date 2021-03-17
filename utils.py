import numpy as np
import math
from scipy import signal

from wifi_preamble import *
from AddChannel import *


def cos_sim(signal1, signal2):
    dot_product = np.dot(signal1,np.conj(signal2))
    l2_signal1 = np.linalg.norm(signal1)
    l2_signal2 = np.linalg.norm(signal2)
    cos_similarity = dot_product # / (l2_signal1 * l2_signal2) #the signals entered normalized 
    return cos_similarity

def NCC(signal1,signal2):
    """
    The normalized cross correlation which boils down to cossine similarity
    """
    n = len(signal1)
    value = np.zeros(2*n-1)+0j
    for i in range(0,len(signal1)):
        value[i]= cos_sim(signal1[:i+1], signal2[-i-1:])
    for j in range(0,len(signal1)-1):
        value[n+j] = cos_sim(signal1[j+1:n], signal2[0:n-1-j])
    return value


def binarySearch(x):
    
    L = 0 
    R = len(x)-1
    while(L < R):
        mid = (L + (R - L)//2 )
        if x[mid] > x[mid+1]:
            L = mid +1
        else:
            R = mid 
    return L 


def startingPointDetector(x):

    k,s=0,0
    window = 50 # Number of decreased samples 
    peak = np.max(x)

    while k < len(x):
        if  0.88*peak <= x[k] <= 0.91*peak:
            begining_point = k
            k = len(x)
        k += 1 
       
    while s < window:
        if  x[s+begining_point] <= x[s+begining_point+1]:
            begining_point = s+begining_point+1
        else:
            s += 1
    return begining_point


def lengthEstimator(x):

    tsh =0.0020
    begining_point = startingPointDetector(x)       
    l = begining_point 
    slope = np.zeros([len(x)-l])
    i = 1

    while i < len(x)-l:
        slope[i-1] = (x[begining_point+i]-x[begining_point])/i
        i += 1 
        
    while True:
        if x[l] <= tsh :
            ending_point_from_tresholding = l
            break
        l += 1

    length_from_tresholding = ending_point_from_tresholding - begining_point

    slope_moving_avg = mov_avg(slope, delt = 5) 
    length_from_binary_search  = binarySearch(slope_moving_avg)
    ending_point_from_binary_search = begining_point + length_from_binary_search   


    starting_point_from_tsh = len(x) - ending_point_from_tresholding 
    starting_point_from_bs = len(x) - ending_point_from_binary_search 



    begining_point = begining_point*10 
    length_from_tresholding, starting_point_from_tsh , ending_point_from_tresholding =length_from_tresholding*10, starting_point_from_tsh*10 , ending_point_from_tresholding*10
    length_from_binary_search, ending_point_from_binary_search,starting_point_from_bs  = length_from_binary_search*10 , ending_point_from_binary_search*10 , starting_point_from_bs*10



    # plt.plot(slope2)
    # plt.figure()
    # plt.plot(x)
    # plt.sho
    # breakpoint()
    return  begining_point, length_from_tresholding, starting_point_from_tsh , ending_point_from_tresholding, length_from_binary_search, ending_point_from_binary_search,starting_point_from_bs


def mov_avg(x,delt):
    new_t = []
    for i in range(0,len(x) ):
        if( len(x[i:delt+i]) % delt == 0):
            suma = sum(x[i:delt+i])/delt
            new_t.append(suma)
        else:
            delt = delt - 1
            suma = sum(x[i:delt+i])/delt
            new_t.append(suma)
    return new_t


def complex_normalize(signals, norm = "l2"):
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


def load_data_from_npz(file_name="data.npz", type= ""):
    """
    Load data from npz file
    """
    data = np.load(file_name)

    # arr_4 is carrier frequency of train
    # arr_5 is carrier frequency of test 
    fc_train = data['arr_4']
    fc_test =  data['arr_5']

    if type == "train":     
        # arr_0 - x train
        # arr_1 - y train

        signals = data['arr_0']
        one_hot_labels = data['arr_1']  # in one hot form
        labels = np.argmax(one_hot_labels, axis=1)

    elif type == "test":
                # arr_2 - x test - signals
                # arr_3 - y test - one hot labels

        signals = data['arr_2']
        one_hot_labels = data['arr_3']  # in one hot form
        labels = np.argmax(one_hot_labels, axis=1)


    return signals, labels,fc_train,fc_test


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


def channelAdder(clustered_signals):

    clustered_signals_with_noise = np.zeros([ len(clustered_signals[:,0,:]),len(clustered_signals[0,:,:]), len(clustered_signals[0,0,:])])
    # complex_signals_random_640 = np.zeros([complex_signals_random[0].shape,complex_signals_random[1].shape/10])

    for i in range(0,clustered_signals.shape[0]):
        for j in range(0,clustered_signals.shape[1]):

            # complex_signals_random_640= complex_signals_random[i,:]
            clustered_signals_with_noise[i,j,:] = add_custom_fading_channel(clustered_signals[i,j,:], snr=500, sampling_rate=20e6, seed = i , beta=None, delay_seed=None,  channel_type=1, channel_method = 'FFT', noise_method = 'reg')
    
    return clustered_signals_with_noise


def preambleDetectionFromDataset(clustered_signals,preamble,length_from_binary_search):

    print('\n Preambles are extracting from the dataset \n')

    signal_preambles = np.zeros([clustered_signals.shape[0], clustered_signals.shape[1], length_from_binary_search ]) + 0j 
    import pdb; pdb.set_trace()
    for i in tqdm(range(0,clustered_signals.shape[0])):
        for j in range(0,clustered_signals.shape[1]):

            corr = np.abs(signal.correlate(clustered_signals[i,j,:],preamble,'full'))
            peak = np.argmax(corr)
            starting_point_for_regarding_signal = peak - length_from_binary_search
            signal_preambles[i,j,:] = clustered_signals[i, j, starting_point_for_regarding_signal:starting_point_for_regarding_signal+length_from_binary_search]
    import pdb; pdb.set_trace()
    return signal_preambles


def mixData(estimated_preambles,labels):
    import pdb; pdb.set_trace()
    remixed_signals = np.zeros([len(labels),estimated_preambles.shape[2],2])
    # remixed_labels = 
    location_of_signal_in_mixture = [] 

    i = 0 

    while i < estimated_preambles.shape[0] - 1:
    
        location_of_signal_in_mixture = np.array(np.where(labels == i))

        for j in range(0,location_of_signal_in_mixture.shape[1]):
            remixed_signals[location_of_signal_in_mixture[:,j],:,0] = np.real(estimated_preambles[i,j,:])
            remixed_signals[location_of_signal_in_mixture[:,j],:,1] = np.imag(estimated_preambles[i,j,:])
        i += 1 

    new_one_hot_labels = np.zeros([3800,19])

    for i in range(0,remixed_signals.shape[0]):

        one_hot_transmitter = np.zeros([estimated_preambles.shape[0]])
        one_hot_transmitter[labels[i]] = 1
        new_one_hot_labels[i,:] = one_hot_transmitter

    import pdb; pdb.set_trace()
    return remixed_signals, new_one_hot_labels 


