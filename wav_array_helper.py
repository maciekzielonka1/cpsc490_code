# This module will help with wav files by reshaping waveform arrays
# into np_arrays, in which each row corresponds to a segment of the desired length
# This module will also create input and output arrays, gathered from different audio files

def split_wav_into_segments(audio_path, segment_len_in_secs):
    # TODO: create np_arrays for a wav_file, in which each entry corresponds to a segment of lenght segment_len_in_secs

def split_wav_into_segments_with_overlap(audio_path, segment_len_in_secs, overlap_len_in_secs):
    # TODO: does the same as the above function, but entries overlap by a length of overlap_len_in_secs

def create_data_and_label_arrays(audio_dir, csv_dir, segment_len_in_secs):
    # TODO: consolidates all the data from audio_dir and csv_dir into np_arrays that will serve as input and output to an svm