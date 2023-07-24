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