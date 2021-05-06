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
            pred = clf.predict(features)
            engagement_tracker.extend(pred)
            current_speaker = speaker
        else:
            if current_speaker == speaker:
                engagement_tracker.extend([-1])
            current_speaker = speaker

    return silent_periods, silent_period_lengths, engagement_tracker


def find_silence_slope(silent_period_lengths, sr = 22050):
    ys = np.array(silent_period_lengths).reshape(-1, 1) / sr
    xs = np.array([i for i in range(len(silent_period_lengths))]).reshape(-1, 1)
    reg = LinearRegression().fit(xs, ys)
    print(reg.coef_)
    plt.scatter(xs, ys)
    plt.title("Duration of Silence Over Time")
    plt.show()
    return reg.coef_[0]

def extract_all_features_from_diarized_interview(y, audio_name, diarization_labels, clf):
    n_chunks = len(diarization_labels)
    num_silent_chunks = 0
    num_child_chunks = 0
    total_child_len = 0
    total_silence_len = 0
    num_adult_chunks = 0
    total_adult_len = 0
    num_engaged_chunks = 0
    num_disengaged_chunks = 0
    duration_engaged = 0
    duration_disengaged = 0
    for label in diarization_labels:
        chunk_start = label[1]
        chunk_end = label[2]
        speaker = label[3]
        chunk_len = chunk_end - chunk_start
        if speaker == 'Silence':
            num_silent_chunks += 1
            total_silence_len += chunk_len
        if speaker == 'Adult':
            num_adult_chunks += 1
            total_adult_len += chunk_len
        if speaker == 'Child':
            num_child_chunks += 1
            total_child_len += chunk_len
            wav_chunk = y[chunk_start:chunk_end]
            features = extract_features_from_chunk(wav_chunk)
            label = clf.predict(features)[0]
            if label == 1:
                num_engaged_chunks += 1
                duration_engaged += chunk_len
            else:
                num_disengaged_chunks += 1
                duration_disengaged += chunk_len
    data = {"n_chunks": n_chunks, 
    "num_silent_chunks": num_silent_chunks, 
    "total_silence_len": total_silence_len, 
    "num_child_chunks": num_child_chunks,
    "total_child_len": total_child_len,
    "num_engaged_chunks": num_engaged_chunks, 
    "duration_engaged": duration_engaged,
    "num_disengaged_chunks": num_disengaged_chunks,
    "duration_disengaged": duration_disengaged,
    "num_adult_chunks": num_adult_chunks,
    "total_adult_len": total_adult_len
    }
    return pd.DataFrame(data, index=[0])

def extract_structural_features_from_wav(audio_path, clf):
    y, sr = librosa.load(audio_path)
    diarization_labels = diarize(y, N_CLUSTERS)
    audio_name = path_leaf(audio_path)
    df = extract_all_features_from_diarized_interview(y, audio_name, diarization_labels, clf)
    return df

def extract_structural_features_from_directory(audio_dir, clf):
    data_frames = []
    for f in os.listdir(audio_dir):
        print("extracting features for", f)
        file_path = os.path.join(audio_dir, f)
        features = extract_structural_features_from_wav(file_path, clf)
        features['audio'] = f
        data_frames.append(features)
    return pd.concat(data_frames)


def plot_labellings(labelling):
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
    plt.show()

