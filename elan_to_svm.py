# This module handles the conversion of annotations of audio files to numpy arrays of binary values
# It will store annotations as csv's, then also be able to extract the csv's into useable dictionaries. 

from pympi import Elan
import glob
import math
import os
import numpy as np

def elan_annotation_to_binary(annotation_data, interval_in_ms):
    """ 
    Given elan-annotated data, and how many milliseconds each entry in the array corresponds to, 
    Create an array of binary labels (0 disengaged, 1 engaged, -1 if no data)
    annotation_data: elan-annotated data in the format of [(annotation_time_start, annotation_time_end, annotation)...]
    interval_in_ms: how many milliseconds each entry in the array corresponds to
    returns - labels: a numpy array of binary labels for each interval of the annotation data
    """
    final_time_stamp = annotation_data[-1][1]
    labels = np.zeros(math.ceil(final_time_stamp/interval_in_ms))
    first_time_stamp = annotation_data[0][0]
    labels[:first_time_stamp//interval_in_ms] = 0
    annotation_index = 0
    for annotation in annotation_data:
        label = 1 if annotation[2] == 'Engaged' else 0
        annotation_start_time = annotation[0]
        annotation_end_time = annotation[1]
        labels[annotation_start_time//interval_in_ms:math.ceil(annotation_end_time/interval_in_ms)] = label
    return labels

def annotations_to_csv(annotations_dir, csv_dir, interval_in_ms):
    """
    Saves the binary labels corresponding to annotations in csv files
    annotations_dir: the directory where the Elan annotations are 
    csv_dir: the target directory for the csv files
    interval_in_ms: how many milliseconds each entry in the csv file labels 
    """
    for file in os.listdir(annotations_dir):
        if file.endswith(".eaf"):
        print("converting", file, "to csv")
        csv_name = os.path.join(csv_dir, file[:-4]) + ".csv"
        file = os.path.join(annotations_dir, file)
        file_elan = Elan.Eaf(file)
        annotation_data = file_elan.get_annotation_data_for_tier("engagement_tier")
        labels_for_annotation = elan_annotation_to_binary(annotation_data, interval_in_ms)
        np.savetxt(csv_name, labels_for_annotation, delimiter=",")

def csv_annotations_to_dict(csv_dir):
    """
    csv_dir: the directory of csv files corresponding to the annotations
    returns: label_arrays: a dictionary where the keys are the name of the csv file, and the values are
            the numpy array with binary labels annotating the data
    """
    label_arrays = {}
    for file in os.listdir(csv_dir):
        csv_name = os.path.join(csv_dir, file)
        labels = np.loadtxt(csv_name, delimiter=",")
        label_arrays[file[:-4]] = labels
    return label_arrays