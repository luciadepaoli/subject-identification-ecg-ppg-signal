import numpy as np
import os
import pandas as pd
from matplotlib.gridspec import GridSpec
from ecgdetectors import Detectors
from scipy.signal import resample
from math import floor
import neurokit2 as nk
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import  timedelta, datetime
from scipy.signal import find_peaks
import scipy
from scipy import signal
import matplotlib as mpl
from matplotlib.transforms import Bbox
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import sklearn.metrics
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import StandardScaler, LabelEncoder
from PIL import Image
import time

# DECODING FUNCTIONS
def convert_array_to_signed_int(data, offset, length):
    return int.from_bytes(
        bytearray(data[offset: offset + length]), byteorder="little", signed=True,
    )

def convert_to_unsigned_long(data, offset, length):
    return int.from_bytes(
        bytearray(data[offset: offset + length]), byteorder="little", signed=False,
    )

def conv_ecg(data):
    ecg_session_data = []
    ecg_session_time = []
    if len(data)>0:
        tmp = data[0]
    else:
        tmp = 0x00
    if tmp == 0x00:
        timestamp = convert_to_unsigned_long(data, 1, 8)
        step = 3
        samples = data[10:] # Check 10 ?
        offset = 0
        while offset < len(samples):
            ecg = convert_array_to_signed_int(samples, offset, step)
            offset += step
            ecg_session_data.extend([ecg])
            ecg_session_time.extend([timestamp])
    return ecg_session_data

def segment_ECG(ecg_signal, fs = 130, word_len = 100):
    detectors = Detectors(fs)
    r_peaks = detectors.two_average_detector(np.squeeze(ecg_signal))
    ecg_matrix = []
    original_len = []
    for i in range(len(r_peaks)-1):
        l = r_peaks[i+1] - r_peaks[i]
        ecg_segment = np.array((ecg_signal[r_peaks[i]:r_peaks[i+1]]).reshape(1,-1)[0])
        original_len.append(len(ecg_segment))
        ecg_word = resample(ecg_segment, word_len)
        ecg_matrix.append(ecg_word)

    ecg_matrix = np.array(ecg_matrix)
    return ecg_matrix, r_peaks, original_len

def segment_PPG(ppg_signal, fs = 100, word_len = 100):
    original_len = []
    ppg_matrix = []
    peaks, _ = find_peaks(np.squeeze(ppg_signal), prominence = 30)#, height = 40, distance = 40)
    for i in range(len(peaks)-1):
        dist = int((peaks[i]-peaks[i+1])*3/4)
        ppg_segment = np.array((ppg_signal[peaks[i]-dist : peaks[i+1]-dist]).reshape(1,-1)[0])
        original_len.append(len(ppg_segment))
        ppg_word = resample(ppg_segment, word_len)
        ppg_matrix.append(ppg_word)
        
    ppg_matrix = np.array(ppg_matrix)
    return ppg_matrix, peaks, original_len

def full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels() 
#    items += [ax, ax.title, ax.xaxis.label, ax.yaxis.label]
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])

    return bbox.expanded(1.0 + pad, 1.0 + pad)

def create_array(df):
    scaled = True
    arr = np.array(df["Signal"])
    arr_numpy = []
    for i in range(len(arr)):
        arr[i] = arr[i].replace("[","")
        arr[i] = arr[i].replace("\n","")
        arr[i] = arr[i].replace("]","")
        float_list = []
        for item in arr[i].split():  
            float_list.append(float(item))
        arr_numpy.append(float_list)
        
    arr_numpy = np.array(arr_numpy)
    if scaled:
        scaler = StandardScaler()
        arr_numpy = scaler.fit_transform(arr_numpy.T).T
    return arr_numpy , np.array(df["Label"])

cpalette = ["#FF2D00", "#0050FF", "#00FF2D","#DD00FF", "#0C0052"]
def clustering(y):
    # Finds position of labels and returns a dictionary of cluster labels to data indices.
    yu = np.sort(np.unique(y))
    clustering = OrderedDict()
    for ye in yu:
        clustering[ye] = np.where(y == ye)[0]
    return clustering

def entropy(c, n_sample):
    # Measures the entropy of a cluster
    h = 0.
    for kc in c.keys():
        p=len(c[kc])/n_sample
        h+=p*np.log(p)
    h*=-1.
    return h

def NMI(y_true, y_pred):
    # Computes Simoized mutual information: where y_true and y_pred are both clustering assignments
    
    w = clustering(y_true)
    c = clustering(y_pred)
    n_sample = len(y_true)

    Iwc = 0.
    for kw in w.keys():
        for kc in c.keys():
            w_intersect_c=len(set(w[kw]).intersection(set(c[kc])))
            if w_intersect_c > 0:
                Iwc += w_intersect_c*np.log(n_sample*w_intersect_c/(len(w[kw])*len(c[kc])))
    Iwc/=n_sample
    Hc = entropy(c,n_sample)
    Hw = entropy(w,n_sample)

    return 2*Iwc/(Hc+Hw)

    from sklearn.cluster import DBSCAN

#cpalette = ["#00A6AA", "#0000A6", "#FF4A46"]
cpalette = ["#FF2D00", "#0050FF", "#00FF2D","#DD00FF", "#0C0052"]

def plotting_ax(X, y, ax):
    # plotting function
    for i, yu in enumerate(np.unique(y)):
        pos = (y == yu)   
        ax.minorticks_on()
        ax.set_facecolor("whitesmoke")
        ax.scatter(X[pos,0], X[pos,1],c=cpalette[i%len(cpalette)],s=8)
def plotting_ax(X, y, ax):
    # plotting function
    for i, yu in enumerate(np.unique(y)):
        pos = (y == yu)   
        ax.minorticks_on()
        ax.set_facecolor("whitesmoke")
        ax.scatter(X[pos,0], X[pos,1],c=cpalette[i%len(cpalette)],s=8)