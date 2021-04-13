# This module will handle the conversion of annotations of audio files to numpy arrays of binary values
# It will store annotations as csv's, then also be able to extract the csv's into useable dictionaries. 

def elan_annotation_to_binary(annotation_data, interval_in_ms):
    # TODO: given elan-annotated data, and how many milliseconds each entry in the array corresponds to, 
    # Create an array of binary labels (0 disengaged, 1 engaged, -1 if no data)

def annotations_to_csv(annotations_dir, csv_dir, interval_in_ms):
    # TODO: given a directory full of elan annotations, convert each file of annotations to csv and save
    # in csv_dir

def csv_annotations_to_dict(csv_dir):
    # TODO: Return a dictionary of {"File": np_array}, where np_array corresponds to the binary labels