import numpy as np
import sys
import codecs
from base64 import b64decode as b64dec
from struct import unpack as unpack
import zlib
import bisect as bs
import xml.etree.cElementTree as et

import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def init_scan(inputfile):

    fn1= inputfile
    scan_time=[]
    spec_m = []
    
    spec_idx = 0
    for event, elem in et.iterparse(fn1, ("start", "end")):
        
        if(elem.tag.endswith('cvParam') and elem.attrib['name']=='time array'): break
        
        if(elem.tag.endswith('cvParam') and elem.attrib['name']=='scan start time' and event=='end'):
            scan_time.append( float(elem.attrib['value']) )
        
        if(elem.tag.endswith("spectrum") or elem.tag.endswith("chromatogram")):
            spec_len=elem.attrib['defaultArrayLength']
            spec_idx=int(elem.attrib['index'])
            #print (spec_idx, spec_len)    ##@@ To be deleted...@@##
    
        elif(spec_idx == 0 and elem.tag.endswith('cvParam')):
            if(elem.attrib['name']=='64-bit float'): 	spec_type='d'
            elif(elem.attrib['name']=='32-bit float'): 	spec_type='f'
            elif(elem.attrib['name']=='zlib compression'): 	spec_comp='zlib'
            elif(elem.attrib['name']=='m/z array'): 	spec_name='mz'
            elif(elem.attrib['name']=='intensity array'): 	spec_name='i'
            elif(elem.attrib['name']=='time array'): 	spec_name='t'
        
        elif(spec_idx == 0 and elem.tag.endswith("binary") and event=='end' ):
            #print("$$$ length/float/comp/type", event, spec_idx, spec_len, spec_type, spec_comp, spec_name)
            unpackedData = []
            base64Data   = elem.text.encode("utf-8")
            decodedData  = b64dec(base64Data)
            decodedData  = zlib.decompress(decodedData)
            fmt = "{endian}{arraylength}{floattype}".format(endian = "<", arraylength=spec_len, floattype=spec_type)
            #unpackedData = unpack(fmt, decodedData)
            #print("Data type/length/example ", spec_name, len(unpackedData), unpackedData[0:2])
            
            if   spec_name=='mz':
                unpackedData = unpack(fmt, decodedData)
                spec_m =list(unpackedData)
            #elif spec_name=='i' : spec_i[spec_idx]=list(unpackedData)

    scan_num = spec_idx + 1

    return scan_num, scan_time, spec_m



def extract_spectrums(inputfile, left_bound, right_bound):
    
    fn1= inputfile
    spec_i=[]
    spec_idx = 0
    
    for event, elem in et.iterparse(fn1, ("start", "end")):
        
        if(elem.tag.endswith('cvParam') and elem.attrib['name']=='time array'): break
#        
#        if(elem.tag.endswith('cvParam') and elem.attrib['name']=='scan start time' and event=='end'):
#            scan_time.append(elem.attrib['value'])

        if(elem.tag.endswith("spectrum") or elem.tag.endswith("chromatogram")):
            spec_len = elem.attrib['defaultArrayLength']
            spec_idx = int(elem.attrib['index'])
            #print (spec_idx)
            
            #if left_bound > spec_idx : continue     ##@@ any problem?? @@##
            if spec_idx >= right_bound : break
        
        elif(elem.tag.endswith('cvParam') and spec_idx >= left_bound ):
            if(elem.attrib['name']=='64-bit float'): 	spec_type='d'
            elif(elem.attrib['name']=='32-bit float'): 	spec_type='f'
            elif(elem.attrib['name']=='zlib compression'): 	spec_comp='zlib'
            elif(elem.attrib['name']=='m/z array'): 	spec_name='mz'
            elif(elem.attrib['name']=='intensity array'): 	spec_name='i'
            elif(elem.attrib['name']=='time array'): 	spec_name='t'
        elif(elem.tag.endswith("binary") and event=='end' and spec_idx >= left_bound and spec_name=='i' ):
            #print("$$$ length/float/comp/type", event, spec_idx, spec_len, spec_type, spec_comp, spec_name)
            unpackedData = []
            base64Data   = elem.text.encode("utf-8")
            decodedData  = b64dec(base64Data)
            decodedData  = zlib.decompress(decodedData)
            fmt = "{endian}{arraylength}{floattype}".format(endian = "<", arraylength=spec_len, floattype=spec_type)
            
            unpackedData = unpack(fmt, decodedData)
            spec_i.append( list(unpackedData) )

    return spec_i


def extract_area(inputfile, pos_rt1, pos_rt2, pos_mz1, pos_mz2):
    
    fn1= inputfile
    area=[]
    spec_idx = 0
    
    for event, elem in et.iterparse(fn1, ("start", "end")):
        
        if(elem.tag.endswith('cvParam') and elem.attrib['name']=='time array'): break
        #
        #        if(elem.tag.endswith('cvParam') and elem.attrib['name']=='scan start time' and event=='end'):
        #            scan_time.append(elem.attrib['value'])
        
        if(elem.tag.endswith("spectrum") or elem.tag.endswith("chromatogram")):
            spec_len = elem.attrib['defaultArrayLength']
            spec_idx = int(elem.attrib['index'])
            #print (spec_idx)
            
            #if left_bound > spec_idx : continue     ##@@ any problem?? @@##
            if spec_idx >= pos_rt2 : break
        
        elif(elem.tag.endswith('cvParam') and spec_idx >= pos_rt1 ):
            if(elem.attrib['name']=='64-bit float'): 	spec_type='d'
            elif(elem.attrib['name']=='32-bit float'): 	spec_type='f'
            elif(elem.attrib['name']=='zlib compression'): 	spec_comp='zlib'
            elif(elem.attrib['name']=='m/z array'): 	spec_name='mz'
            elif(elem.attrib['name']=='intensity array'): 	spec_name='i'
            elif(elem.attrib['name']=='time array'): 	spec_name='t'
        elif(elem.tag.endswith("binary") and event=='end' and spec_idx >= pos_rt1 and spec_name=='i' ):
            #print("$$$ length/float/comp/type", event, spec_idx, spec_len, spec_type, spec_comp, spec_name)
            unpackedData = []
            base64Data   = elem.text.encode("utf-8")
            decodedData  = b64dec(base64Data)
            decodedData  = zlib.decompress(decodedData)
            fmt = "{endian}{arraylength}{floattype}".format(endian = "<", arraylength=spec_len, floattype=spec_type)
            
            unpackedData = unpack(fmt, decodedData)
            spec_i = np.array(list(unpackedData))
            area.append( spec_i[pos_mz1: pos_mz2] )
    
    return area

def plot_area(area, mz0, rt0):
    
    im = plt.imshow(area, interpolation='bilinear', cmap='jet', aspect='auto' )
    plt.xlabel('M/Z')
    plt.ylabel('Time')
    plt.title("M/Z: " + str(mz0)+"  RT: " +str(rt0) )
    plt.colorbar()
    #plt.show()
    plt.savefig('PK-'+ str(mz0) + '-'+ str(rt0)  +'.png')
    plt.clf()





