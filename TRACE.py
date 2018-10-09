import os
import math
import numpy as np
from scan_cwt_1 import scan_mp
from getImage_2 import get_image
from predict_3 import predict
import multiprocessing as mp


if __name__ == '__main__':
    
    num_cores = mp.cpu_count()   ## Count the cores of the PC.
    print ('Number of cores detected in this PC:', num_cores)

    NUM_C = max(num_cores-2, 1)   ## MP use (all-2) threads by default.

    Big_RAM = 1   ## See if the RAM of PC is big enough (> 8 times bigger than profile file size)

    K_MEANS = 8  ## Or some interger (2~10 recommended); for k-means clustering of signal images

    window_mz = 6  # the m/z range is 6 points (on both sides)
    window_rt = 30  # The time range is 30 points (on both sides)


    RESULTS_PATH = "./Results"
    if not os.path.isdir(RESULTS_PATH):   ## Will create a folder for results if not existent.
        os.makedirs(RESULTS_PATH)

    ## First step: Preprocessing and initial scanning.
    pks_initial =  scan_mp( '../Data/Centroid-ORG.mzML', RESULTS_PATH = RESULTS_PATH, NUM_C = NUM_C )  ##

    ## Second step: Signal image evaluation.
    images = get_image( '../Data/Profile-ORG.mzML', pks_initial, RESULTS_PATH, Big_RAM, window_mz, window_rt)

    pks_final = predict(images, pks_initial, RESULTS_PATH = RESULTS_PATH, K_means = K_MEANS )


    print ('Done! Final results in ' +  RESULTS_PATH + ' folder.')

