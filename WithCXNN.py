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
import os


from wifi_preamble import ideal_preamble_generation
import lengthFinder as lengthFinder
from lengthFinder import *
from AddChannel import *
from utils import *
import pcaPreambleEstimator as preamble_estimator

 


def main():
    
	val_num_aligned_signals = int(input("\n Enter the number of signals: "))
	
	BASE_PATH = "/home/arinc/pyt_wrkplc/wireless-fingerprinting/data"
	
	estimated_preamble_train, labels_train , clustered_signals_train,length_from_binary_search = preamble_estimator.main(step_size = 10, num_aligned_signal = val_num_aligned_signals, ind_reference_transmitter=18, ind_reference_signal=199,type = "train")
	
	signal_preambles_train =  preambleDetectionFromDataset(clustered_signals_train, estimated_preamble_train, length_from_binary_search)
	
	signals_test, labels_test,fc_train,fc_test = load_data_from_npz(file_name="data.npz",type = "test" )
	comp_signals = signals_test[:, :, 0] + 1j*signals_test[:, :, 1]
	comp_signals = complex_normalize(comp_signals)
	signals_test[:, :, 0] = comp_signals.real
	signals_test[:, :, 1] = comp_signals.imag
	
	clustered_signals_test = cluster_signals(signals=signals_test, labels=labels_test)  # complex valued
	
	signal_preambles_test = preambleDetectionFromDataset(clustered_signals_test, estimated_preamble_train, length_from_binary_search)
	
	mixed_preambles_test,new_one_hot_labels_test =  mixData(signal_preambles_test,labels_test)
	
	
	np.savez(os.path.join(BASE_PATH,'simulations.npz'),mixed_preambles_train,new_one_hot_labels_train,mixed_preambles_test,new_one_hot_labels_test,mixed_preambles_test,new_one_hot_labels_test,fc_train,fc_test,fc_test)
	    

	return signal_preambles_train, mixed_preambles_train, new_one_hot_labels_train,fc_test,fc_train,signal_preambles_test, mixed_preambles_test, new_one_hot_labels_test

if __name__ == '__main__':
    main()
