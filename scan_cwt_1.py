
import numpy as np
import sys
import codecs
from base64 import b64decode as b64dec
from struct import unpack as unpack
import zlib
import bisect as bs
import xml.etree.cElementTree as et

from scipy.signal import convolve
from scipy.stats import scoreatpercentile
import scipy.stats as stats
import scipy.signal as signal
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
import multiprocessing as mp

# https://stackoverflow.com/questions/21027477/joblib-parallel-multiple-cpus-slower-than-single

################### Important Paramenters To Be Changed ###################

mz_min = 25.000        # The minimum of m/z to be evaluated
mz_max = 550.01        # The maximum of m/z to be evaluated
mz_r  = 0.0050         # The m/z bin for signal detection and evaluation (window is +/- this value)

ms_freq = 2       ## The scanning frequency of MS: spectrums/second. Change accordingly
################### Important Paramenters To Be Changed ###################

################### Important Paramenters ###################
min_len_eic = 6   ## Minimum length of a EIC to be scanned by CWT
widths=np.asarray( [i for i in range(1,int(10*ms_freq),1)] + [int(20*ms_freq)] )
gap_thresh = np.ceil(widths[0])
window_size = 30
min_length  = int(len(widths)*0.2)  # org: 0.25
min_snr = 4  # org: 8. This is the Signal Noise Ratio for the wavelet and may needed to be adjusted.
perc=90

############################################
Pick_mlist = np.arange(mz_min,mz_max, mz_r)
max_distances = widths/4.0
max_scale_for_peak = 18
hf_window=int(0.5*window_size)
################### Important Paramenters ###################


############################### ricker ####################################
def ricker(points, a):
    A = 2 / (np.sqrt(3 * a) * (np.pi**0.25))
    wsq = a**2
    vec = np.arange(0, points) - (points - 1.0) / 2
    xsq = vec**2
    mod = (1 - xsq / wsq)
    gauss = np.exp(-xsq / (2 * wsq))
    total = A * mod * gauss
    return total

################################ cwt ######################################
def cwt(data, widths):
    output = np.zeros([len(widths), len(data)])
    for ind, width in enumerate(widths):
        wavelet_data = ricker(min(10 * width, len(data)), width)
        #print("width, wavelet, data ", width, len(wavelet_data), len(data))
        output[ind, :] = convolve(data, wavelet_data, mode='same')
    return output

########################### identify_ridge_lines ###############################
def identify_ridge_lines(matr, max_distances, gap_thresh, min_length):
    
    if(len(max_distances) < matr.shape[0]):
        raise ValueError('Max_distances must have at least as many rows as matr')

    all_max_cols = boolrelextrema(matr, np.greater, axis=1, order=1, mode='clip')
    has_relmax = np.where(all_max_cols.any(axis=1))[0]
    if(len(has_relmax) == 0): return []

    start_row = has_relmax[-1]
    ridge_lines = [[[start_row],
                    [col],
                    0] for col in np.where(all_max_cols[start_row])[0]]
    final_lines = []
        
    rows = np.arange(start_row - 1, -1, -1)
    cols = np.arange(0, matr.shape[1])
    
    for row in rows:
        this_max_cols = cols[all_max_cols[row]]
            
        for line in ridge_lines:
            line[2] += 1
            
        prev_ridge_cols = np.array([line[1][-1] for line in ridge_lines])
        for ind, col in enumerate(this_max_cols):
            line = None
            if(len(prev_ridge_cols) > 0):
                diffs = np.abs(col - prev_ridge_cols)
                closest = np.argmin(diffs)
                if diffs[closest] <= max_distances[row]:
                    line = ridge_lines[closest]
                
            if(line is not None):
                line[0].append(row)
                line[1].append(col)
                line[2] = 0
            else:
                new_line = [[row],[col],0]
                ridge_lines.append(new_line)
                    
        for ind in range(len(ridge_lines) - 1, -1, -1):
            line = ridge_lines[ind]
            if line[2] > gap_thresh:
                final_lines.append(line)
                del ridge_lines[ind]

    out_lines = []
    for line in (final_lines + ridge_lines): 
        sortargs = np.array(np.argsort(line[0]))
        rows, cols = np.zeros_like(sortargs), np.zeros_like(sortargs)
        rows[sortargs] = line[0]
        cols[sortargs] = line[1]
        out_lines.append([rows, cols])

    return out_lines


############################# boolrelextrema ###################################
def boolrelextrema(data, comparator, axis=0, order=1, mode='clip'):
    
    #print("order ", order)
    
    if((int(order) != order) or (order < 1)):
        raise ValueError('Order must be an int >= 1')
    
    datalen = data.shape[axis]
    locs = np.arange(0, datalen)
    
    results = np.ones(data.shape, dtype=bool)
    
    main = data.take(locs, axis=axis, mode=mode)
    for shift in range(1, order + 1):
        plus = data.take(locs + shift, axis=axis, mode=mode)
        minus = data.take(locs - shift, axis=axis, mode=mode)
        results &= comparator(main, plus)
        results &= comparator(main, minus)
        
        if(~results.any()):
            return results
    return results


############# Merge the signal list: delete the repeated ones ##########
def merge(f):
    m_r = 0.005   # Uncertainty for m/z allowed.
    t_r = 0.1    # Uncertainty for rt  allowed.
    cnew=0
    
    f2 = []
    for i in range(len(f)-1):
        bb=0
        for j in range(i+1,len(f)):
            if abs(f[i][0] - f[j][0] )< m_r and abs(f[i][1] - f[j][1]) < t_r :
                bb=bb+1
                break
        if bb < 1 :
            cnew = cnew + 1
            f2.append(f[i])
    f2.append(f[-1])
    return f2


############# Scan EICs ###################
def scan_EIC(ind):
    
    #global pks_found
    global spec_m
    global spec_i
    global scan_time
    
    pks_found = []
    
    scantime_min = float(scan_time[0])
    scantime_max = float(scan_time[len(scan_time)-1])
    
    mz_t = round( float(Pick_mlist[ind]), 3)
    chrm_ht=[];   chrm_mz=[];   chrm_tt=[];     chrm_ct=[]
    
    mz1=mz_t-mz_r
    mz2=mz_t+mz_r
    
    for t in range(len(scan_time)):    # range(spec_idx+1)
        pos1=bs.bisect_left(spec_m[t], mz1)
        pos2=bs.bisect_left(spec_m[t], mz2)
        
        ht_max=0.0
        mz_max=-100
        if pos2-pos1:
            for k in range(pos1, pos2, 1):
                if spec_i[t][k] > ht_max:
                    ht_max=spec_i[t][k]
                    mz_max=spec_m[t][k]-mz_t
            #print "largest one ", mz_max, ht_max
            chrm_ht.append(ht_max)
            chrm_mz.append(mz_max)
            chrm_tt.append( round( float(scan_time[t]), 3) )
            chrm_ct.append(pos2-pos1)


    if len(chrm_ht) < min_len_eic:    #continue
        return

    cwtmatr=cwt(np.asarray(chrm_ht), widths)

    ######################
    # Ridges by CWT
    ######################
    
    ridge_lines = identify_ridge_lines(cwtmatr, max_distances, gap_thresh, min_length)
    
    # Gathering Information from CWT
    # row_one: the smallest scale
    # row_inf: the largest scale
    num_points = cwtmatr.shape[1]
    
    row_one = cwtmatr[0, :]
    row_inf = cwtmatr[cwtmatr.shape[0]-1, :]
    
    peak_loc=[]
    peak_data=[]
    peak_info_snr=[]
    peak_info_shape=[]

    max_pts=[]

    for line in ridge_lines:
        
        #first pass (too close to the edges)
        if chrm_tt[line[1][0]] < scantime_min + 0.5 or chrm_tt[line[1][0]] > scantime_max - 0.5: continue
        
        #second pass (min_lenght)
        if len(line[0]) < min_length: continue

        tm=chrm_tt[line[1][0]]
        tm1=tm-0.5
        tm2=tm+0.5
        pk1=bs.bisect_left(chrm_tt, tm1)
        pkm=bs.bisect_left(chrm_tt, tm)
        pk2=bs.bisect_left(chrm_tt, tm2)
        if pkm-pk1 < 3 : continue
        if pk2-pkm < 3 : continue
        
        if chrm_tt[pkm + 2] - chrm_tt[pkm - 3] > 1.5 * 5* 0.503/60 : continue
        
        pkk1=max(0, pkm-hf_window)
        pkk2=min(pkm+hf_window,len(chrm_ht)-1)
        #cwtmatr[len(widths)-1][pkk1:pkk2]
        
        line_len = len(line[1])
        max_row = line[0][line_len-1]
        max_cm = line[1][line_len-1]
        
        slope1 = (cwtmatr[max_row][pkm] - cwtmatr[max_row][pkk1])/(pkm-pkk1)
        slope2 = (cwtmatr[max_row][pkk2] - cwtmatr[max_row][pkm])/(pkk2-pkm)
        if 0.2 < slope1/slope2 < 5 : continue
        
        line_val=[]
        line_scl=[]
        for i in range(len(line[0])):
            if widths[line[0][i]] < max_scale_for_peak:
                line_val.append( cwtmatr[line[0][i], line[1][i]] )
                line_scl.append( widths[line[0][i]] )

        if len(line_val) == 0: continue
        
        line_val_opt = max(line_val)
        line_val_ave = np.mean(line_val)
        line_scale_opt = line_scl[np.argmax(line_val)]

        # snr (using the beginning of the line to locate the time location)
        ind=line[1][0]
        window_start = max(ind - hf_window, 0)
        window_end = min(ind + hf_window, num_points)
        
        
        noises =stats.scoreatpercentile(abs(row_one[window_start:window_end]), perc)
        line_val=[]
        for i in range(window_start, window_end):
            if row_one[i] < noises: line_val.append(row_one[i])
        means  =np.mean(line_val)
        stdevs =np.std(line_val)
        
        data_noises=stats.scoreatpercentile(chrm_ht[window_start: window_end], perc)
        data_means=np.mean(chrm_ht[window_start: window_end])
        data_stdevs=np.std(chrm_ht[window_start: window_end])
        
        snr1 = line_val_ave / stdevs
        snr2 = line_val_opt / stdevs
        snr3 = row_one[ind] / stdevs
        snr4 = cwtmatr[2, ind]/stdevs

        if snr1 > min_snr and snr4 > 1.5:
            
            asym1=row_inf[ind] - row_inf[window_start]
            asym2=row_inf[ind] - row_inf[window_end-1]
            
            if asym1 < 0 or asym2 < 0:
                if snr1 < 10 :                           ### org: 20
                    continue

            pk = line[1][0]

            line_len = len(line[1])
            max_row = line[0][line_len-1]
            max_cm = line[1][line_len-1]
            
            
            peak_loc.append(pk)
            max_pts.append([max_row, max_cm])
                
            clist = [ col[pk] for col in cwtmatr]
            clist.sort()
            cmax = round(clist[len(clist)-1], 1)
            #cmax = round( float(cwtmatr[max_row][max_cm]), 1)

            mz_real = round( float( mz_t + chrm_mz[pk]),3 )  ## Record the real m/z value

            pks_found.append([ mz_real, chrm_tt[pk], chrm_ht[pk], cmax, snr1 ])
            #fout.write('%s  %s  %s  %s  %s  %s\n' % (mz_t, mz_real, chrm_tt[pk], chrm_ht[pk], cmax, snr1) )

            print  (mz_t, mz_real, chrm_tt[pk], chrm_ht[pk], cmax, snr1)

    return np.array(pks_found)

############# Scan EICs ###################


spec_m = [ ]    # empty matrix for m/z
spec_i = [ ]    # empty matrix for intensity
scan_time=[]

#pks_found = []
############################## Initial scan ############################################
def scan_mp( centroid_file_mzML, RESULTS_PATH, NUM_C = 1 ):
    
    global spec_m
    global spec_i
    global scan_time
    
    fn1= centroid_file_mzML   # File name to be changed. Remember to specify the folder location for this file.
    fout = RESULTS_PATH+ "/Initial_pks.txt" # File for save the initial scaning results. To be changed
    
    spec_comp = []
    for event, elem in et.iterparse(fn1, ("start", "end")):
        if(elem.tag.endswith('cvParam') and elem.attrib['name']=='time array'): break
        
        if(elem.tag.endswith('cvParam') and elem.attrib['name']=='scan start time' and event=='end'):
            scan_time.append(elem.attrib['value'])
        
        if(elem.tag.endswith("spectrum") or elem.tag.endswith("chromatogram")):
            spec_len=elem.attrib['defaultArrayLength']
            spec_idx=int(elem.attrib['index'])
        elif(elem.tag.endswith('cvParam')):
            if(elem.attrib['name']=='64-bit float'): 	spec_type='d'
            elif(elem.attrib['name']=='32-bit float'): 	spec_type='f'
            elif(elem.attrib['name']=='zlib compression'): 	spec_comp='zlib'
            elif(elem.attrib['name']=='m/z array'): 	spec_name='mz'
            elif(elem.attrib['name']=='intensity array'): 	spec_name='i'
            elif(elem.attrib['name']=='time array'): 	spec_name='t'
        
        elif(elem.tag.endswith("binary") and event=='end'):
            unpackedData = []
            
            if (elem.text):    ## Sometimes the spectrum can be empty.
                base64Data   = elem.text.encode("utf-8")
                decodedData  = b64dec(base64Data)
                if spec_comp =='zlib':
                    decodedData  = zlib.decompress(decodedData)   ## Do this only if the file is compressed.
                
                fmt = "{endian}{arraylength}{floattype}".format(endian = "<", arraylength=spec_len, floattype=spec_type)
                unpackedData = unpack(fmt, decodedData)
                #print("Data type/length/example ", spec_name, len(unpackedData), unpackedData[0:2])
                
                if   spec_name=='mz':
                    spec_m.append( list(unpackedData) )
                elif spec_name=='i' :
                    spec_i.append( list(unpackedData) )

    
    pks_found_all = Parallel(n_jobs=(NUM_C))(delayed(scan_EIC)(ind)for ind in range(len(Pick_mlist)) )
    
    pks_merged = []
    for jj in range(np.shape(pks_found_all)[0]):
        ttt = np.array(pks_found_all[jj])
        if ttt.any():
            pks_merged.extend(ttt)


    ############## Done! ################
    print (np.shape(pks_merged))
    
    pks_final = merge(pks_merged)

    np.savetxt(fout, pks_final, fmt='%.4f', delimiter="  ")
    print ('Initial screening done! Total peaks found: ', len(pks_final) )

    return pks_final

