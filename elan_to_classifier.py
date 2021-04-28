# This module handles the conversion of annotations of audio files to numpy arrays of binary values
# It will store annotations as json files, with keys corresponding to segments of wav arrays, and 
# values corresponding to binary annotations for those segments. 
from pympi import Elan
import glob
import math
import os
import numpy as np
import json
import librosa
from helpers import path_leaf

def annotations_to_json(eaf_dir, json_dir):
    """
    Saves the binary labels corresponding to annotations in json files
    eaf_dir: the directory where the Elan annotations are 
    json_dir: the target directory for the json files
    """
    for file in os.listdir(eaf_dir):
        if file.endswith(".eaf"):
            print("converting", file, "to json")
            file_name = os.path.join(json_dir, file[:-4]) + ".json"
            file = os.path.join(eaf_dir, file)
            file_elan = Elan.Eaf(file)

            # Get all the data under the engagement_tier tier
            annotation_data = file_elan.get_annotation_data_for_tier("engagement_tier")
            labels_for_annotation = elan_annotation_to_binary(annotation_data)

            # Create a json file storing the dictionary of {"timeslot1,timeslot2": 0/1(engaged/disengaged)}
            j = json.dumps(labels_for_annotation)
            f = open(file_name, "w")
            f.write(j)
            f.close()

def elan_annotation_to_binary(annotation_data):
    """ 
    Given elan-annotated data create a dictionary of binary labels (0 disengaged, 1 engaged)
    annotation_data: elan-annotated data in the format of [(annotation_time_start, annotation_time_end, annotation)...]
    returns - label_dict: a dictionary in which the keys are the start and end times of a chunk, 
            separated by a comma, and the values are 0 or 1
    """
    label_dict = {}
    for annotation in annotation_data:
      label = 1 if annotation[2] == 'Engaged' else 0
      label_dict["{0},{1}".format(annotation[0], annotation[1])] = label
    return label_dict

def save_all_chunks_with_labels(audio_dir, json_dir, csv_dir):
    """
    A helper function that loops through all the available json files with annotations,
    and save the corresponding chunks of audio as csv files, along with the binary label for engagement
    audio_dir: the path to the directory with ".wav" files
    json_dir: the path to the directory with the saved ".json" files
    csv_dir: the path to the directory in which to save ".csv" files  
    """
    for file in os.listdir(json_dir):
        file_path = os.path.join(json_dir, file)
        audio_file_path = os.path.join(audio_dir, file)[:-4] + "wav"
        with open(file_path) as f:
            data = json.load(f)
            save_arrays_with_labels(audio_file_path, data, csv_dir)

def save_arrays_with_labels(audio_path, annotation_dict, csv_dir):
    """
    Saves each chunk of an individual ".wav" file to a ".csv" file, with the last entry being the binary label
    audio_path: the path to a ".wav" file
    annotation_dict: a dictionary where the keys contain the start and end slices of a desired chunk, and the
            value is the binary annotation for that chunk
    csv_dir: the directory where the ".csv" files are to be stored
    """
    y, sr = librosa.load(audio_path)
    for key, value in annotation_dict.items():
        key_start, key_end = key.split(",")
        key_start = (int)(key_start)
        key_end = (int)(key_end)
        segment_start = (int)((key_start/1000) * sr)
        segment_end = (int)((key_end/1000) * sr)
        wav_segment = y[segment_start:segment_end]
        fname = os.path.join(csv_dir, path_leaf(audio_path))[:-4] + "_" + key + ".csv"
        arr = np.append(wav_segment, value)
        np.savetxt(fname, arr, delimiter=",")