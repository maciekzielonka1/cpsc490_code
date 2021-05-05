# Copied from https://github.com/rramnauth2220/audio-feature-extraction.git

# file support
import os
from os import listdir
from os.path import isfile, join, basename, splitext
import librosa

##################################################
#              AUXILIARY FUNCTIONS               #
##################################################

def remove_extension(file):
    return os.path.splitext(file)[0]

def path_leaf(path):
    head, tail = os.path.split(path)
    return tail or os.path.basename(head)

def get_files(directory, valid_exts):
    return ([directory + x for x in 
        [f for f in listdir(directory) if (isfile(join(directory, f)) # get files only
        and bool([ele for ele in valid_exts if(ele in f)])) ]         # with valid extension
        ])  

def generate_file_name(outDir, inFile, lowcut, highcut, suffix = "_filtered", extension = ".wav"):
    return outDir + remove_extension(path_leaf(inFile)) + '_' + str(int(lowcut)) + '-' + str(int(highcut)) + suffix + extension

def convertMillis(millis):
    """
    Function to convert milliseconds into `MIN:SEC` Format
    """
    millis = int(millis)
    seconds=(millis/1000)%60
    seconds = int(seconds)
    minutes=(millis/(1000*60))%60
    minutes = int(minutes)
    hours=(millis/(1000*60*60))%24
    return ("%d:%d" % (minutes, seconds))

def convertSecs(seconds, half_seconds=False):
    """
    function to convert seconds into `MIN:SEC` format
    """
    if half_seconds:
        seconds /= 2.0
    minutes=(seconds//60)%60
    minutes = int(minutes)
    frac, whole = math.modf(seconds)
    seconds = whole % 60
    seconds += frac
    return ("%d:%.1f" % (minutes, seconds))