import pandas as pd
import librosa
import numpy as np
import os
from speaker_diarization import diarize
from feature_extractors import extract_features_from_chunk
from helpers import path_leaf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

N_CLUSTERS = 3


def extract_structural_features_from_diarized_interview(y, diarization_labels, clf):
    """
    Extracts structural features from a diarized interview, which could then be used to 
    help analyze the social contingency of an interaction

    y - a waveform array
    diarization_labels - a list of [(`speaker_id`, `segment_start_frame`, `segment_end_frame`, `speaker_label`)...]
    clf - an sklearn classifier
    returns:
    silent_periods - a list of all the silent segments in y, listed as [(`silence_start_frame`, `silence_end_frame`)]
    silent_period_lengths - a list of the lengths (in frames) of each segment of silence
    engagement_tracker - a list of:
        -1 if after a period of silence the adult speaks again
        0 if a child speaks but the classifier classifies them as disengaged
        1 if a child speaks and the classifier classifies the segment as engaged
        `silence_len` the duration of silence if the segment is a silent segment
    """
    silent_periods = []
    silent_period_lengths = []
    current_speaker = None
    engagement_tracker = []
    for label in diarization_labels:
        start_time = int(label[1])
        end_time = int(label[2])
        speaker = label[3]
        if speaker == 'Silence':
            silent_periods.append((start_time, end_time))
            silent_period_lengths.extend([end_time - start_time])
            engagement_tracker.extend([end_time - start_time])
        elif speaker == "Child":
            chunk = y[start_time:end_time]
            features = extract_features_from_chunk(chunk)
            probs = clf.predict_proba(features)[0]
            pred = int(clf.predict(features)[0])
            engagement_tracker.extend([(pred, probs[pred])])
            current_speaker = speaker
        else:
            if current_speaker == speaker:
                engagement_tracker.extend([-1])
            current_speaker = speaker

    return silent_periods, silent_period_lengths, engagement_tracker


def find_silence_slope(silent_period_lengths, sr = 22050):
    """
    Uses linear regression to determine the slope of silence duration over time
    silent_period_lengths - a list of durations of silence (in frames) of a conversation 
    """
    ys = np.array(silent_period_lengths).reshape(-1, 1) / sr
    xs = np.array([i for i in range(len(silent_period_lengths))]).reshape(-1, 1)
    reg = LinearRegression().fit(xs, ys)
    print(reg.coef_)
    plt.scatter(xs, ys)
    plt.title("Duration of Silence Over Time")
    plt.show()
    return reg.coef_[0]

def plot_labellings(labelling):
    """
    A tool that might prove useful, plots the diarization labels over time
    """
    y = []
    x = []
    label_dict = {"Silence": 2, "Adult": 1, "Child": 0}
    for label in labelling:
        lbl = label[3]
        lbl_num = label_dict[lbl]
        lbl_start = label[1]
        lbl_end = label[2]
        x.extend([lbl_start, lbl_end])
        y.extend([lbl_num, lbl_num])

    plt.plot(x, y)
    plt.title("Diarization Over Time")
    plt.show()

