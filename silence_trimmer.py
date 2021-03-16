from pydub import AudioSegment
import sys
import os
from os import path, listdir
from os.path import isdir
from helpers import *
from listen import remove_leading_silence

SILENCE_THRESHOLD = -35.0
TRIMMED_SILENCE_DIR_NAME = "trimmed_silence/"

def remove_silence_from_directory(directory):
    for item in listdir(directory):
        path_name = directory + item
        print("removing silence from directory", path_name)
        new_folder = path_name + '/' + TRIMMED_SILENCE_DIR_NAME
        if not os.path.exists(new_folder):
            os.mkdir(new_folder)
            print("mkdir", new_folder)
        for wav in listdir(path_name):
            wav_path = path_name + '/' + wav
            if isdir(wav_path):
                continue
            print("trimming ", wav)
            clip = None
            try:
                clip = AudioSegment.from_file(wav_path, format='wav')
            except:
                continue
            start_trim = remove_leading_silence(clip, silence_threshold=SILENCE_THRESHOLD)
            end_trim = remove_leading_silence(clip.reverse(), silence_threshold=SILENCE_THRESHOLD)
            clip_duration = clip.duration_seconds * 1000
            trimmed_clip = clip[start_trim:(clip_duration)-end_trim]
            trimmed_clip_duration = len(trimmed_clip)
            trimmed_clip_name = "{0}{1}_trimmed.wav".format(new_folder, remove_extension(wav))
            if trimmed_clip_duration > 1000:
                print("generating trimmed clip", trimmed_clip_name)
                trimmed_clip.export(trimmed_clip_name, format='wav')

if __name__ == "__main__":
    # Take as input the directory with all the wav_files
    directory = sys.argv[1]
    remove_silence_from_directory(directory)
    

