# This module wasn't really used in the final stages of the project, more the earlier stages when 
# I was trying some things out, but it could be helpful if anyone ever needs to reshape `.wav` files 
# into segments where each row corresponds to a segment of a desired length
# 
# This module will help with wav files by reshaping waveform arrays
# into np_arrays, in which each row corresponds to a segment of the desired length
# This module will also create input and output arrays, gathered from different audio files

import numpy as np
import math
from resemblyzer import  preprocess_wav, VoiceEncoder
import librosa
from elan_to_svm import csv_annotations_to_dict
import os

def split_wav_into_segments(audio_path, segment_len_in_secs):
    """ 
    Reshapes a wav file into a numpy array where each row corresponds to a segment of length segment_len_in_secs
    audio_path: path to the desired wav file
    segment_len_in_secs: how long each entry in the array should be in seconds
    returns - reshaped_arr: the reshaped numpy array
    """
    y, sr = librosa.load(audio_path)
    y_size = y.shape[0]
    samples_per_segment = (int)(math.ceil(sr * segment_len_in_secs))
    n_segments = y_size / samples_per_segment
    n_rows = (int)(math.ceil(n_segments))
    y_final_size = n_rows * samples_per_segment
    n_zeros = (int)(y_final_size - y_size)
    padded_arr = np.pad(y, (0, n_zeros))
    reshaped_arr = padded_arr.reshape(n_rows, samples_per_segment)
    print(reshaped_arr.shape)
    return reshaped_arr

def create_data_and_label_arrays(audio_dir, csv_dir, segment_len_in_secs):
    """
    consolidates desired numpy arrays into two big arrays, which will correspond to the inputs and labels for an svm
    audio_dir: the directory containing the .wav files
    csv_dir: the directory containing the .csv files with annotations
    segment_len_in_secs: how long (in seconds), each entry in the arrays corresponds to
    returns - input_arr: a numpy array with all the entries
            label_arr: a numpy array with the labels, the indices of which correspond to the indices of input_arr
    """
    label_array_dict = csv_annotations_to_dict(csv_dir)
    first_row = True
    input_arr = None
    labels_arr = None
    for file_name in label_array_dict.keys():
        audio_file = os.path.join(audio_dir, file_name) + ".wav"
        wav_arr = split_wav_into_segments(audio_file, segment_len_in_secs)
        labels = label_array_dict[file_name]
        if first_row:
            input_arr = wav_arr
            labels_arr = labels
            first_row = False
        else:
            input_arr = np.append(input_arr, wav_arr, 0)
            labels_arr = np.append(labels_arr, labels, 0)
    print(input_arr.shape, labels_arr.shape)
    return input_arr, labels_arr

def split_wav_into_segments_with_overlap(audio_path, segment_len_in_secs, overlap_len_in_secs):
    """
    Does the same as `spilt_wav_into_segments`, but each entry overlaps with the previous entry by 
    a length of `overlap_len_in_secs` seconds
    """
    res_arr = []
    y, sr = librosa.load(audio_path)
    y_size = y.shape[0]
    samples_per_segment = (int)(math.ceil(sr * segment_len_in_secs))
    n_segments = y_size / samples_per_segment
    n_rows = (int)(math.ceil(n_segments))
    y_final_size = n_rows * samples_per_segment
    n_zeros = (int)(y_final_size - y_size)
    padded_arr = np.pad(y, (0, n_zeros))
    res_arr = padded_arr[0:samples_per_segment].reshape(1, samples_per_segment)
    i = math.ceil(samples_per_segment*overlap_len_in_secs)
    while i + samples_per_segment <= y_final_size:
        next_seg = padded_arr[i:i+samples_per_segment].reshape(1, samples_per_segment)
        res_arr = np.append(res_arr, next_seg, 0)
        i += math.ceil(samples_per_segment*overlap_len_in_secs) 
    return res_arr