import sys
import numpy as np
import bisect as bs
import mzmlReadRaw as read

import matplotlib.pyplot as plt
import matplotlib.cm as cm


def get_image( profile_file_mzML, pk_list, RESULTS_PATH, Big_RAM = 0, window_mz = 6, window_rt = 30 ):   ## Add a choice of big RAM or not ###

    inputfile = profile_file_mzML   ##@@ To be changed @@##

    [scan_num, scan_t, mz_list] = read.init_scan(inputfile)

    if (len(scan_t) > 2 and scan_t[1] - scan_t[0] > 0.1):  ## if gap >0.1: second..
        scan_t = [i/60.0 for i in scan_t]

    if Big_RAM > 0:
        ht_matrix = read.extract_spectrums(inputfile, 0, len(scan_t))
        ## Store the whole intensity matrix. May only work for PC with big RAM.

    images = []

    for i in range(len(pk_list)):
    #for i in range(3):

        if i%50 ==0:
            print ('Extract image of signal NO.: ', i)

        mz0 = float(pk_list[i][0])
        rt0 = float(pk_list[i][1])
        
        pos_mz=bs.bisect_left(mz_list, mz0)
        pos_mz1 = pos_mz - window_mz
        pos_mz2 = pos_mz + window_mz
        if pos_mz2 >= len(mz_list):
            pos_mz1 = len(mz_list)- window_mz*2
            pos_mz2 = len(mz_list)
        elif pos_mz1 < 0:
            pos_mz1 = 0
            pos_mz2 = 2*window_mz
        
        pos_rt=bs.bisect_left(scan_t, rt0)
        pos_rt1 = pos_rt-window_rt
        pos_rt2 = pos_rt+window_rt
        if pos_rt2 >= len(scan_t):
            pos_rt1 = len(scan_t)- window_rt*2
            pos_rt2 = len(scan_t)
        elif pos_rt1 < 0:
            pos_rt1 = 0
            pos_rt2 = 2*window_rt

        area = []
        if Big_RAM > 0:
            for t in range(pos_rt1, pos_rt2):
                htgrids = ht_matrix[t]
                grids_part = [htgrids[i] for i in range(pos_mz1, pos_mz2)]
                area.append(grids_part)
                # print (i, np.shape(area))
        else:
            area = read.extract_area(inputfile, pos_rt1, pos_rt2, pos_mz1, pos_mz2)

        images.append(np.reshape(area, window_mz*window_rt*4))

    print (np.shape(images))
    f2 = ( RESULTS_PATH+ "/Images_pks.txt")   ##@@ To be changed @@##
    np.savetxt(f2, images, fmt='%.1f',delimiter=' ')

    return images





