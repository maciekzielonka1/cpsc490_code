import numpy as np
import time
import essentia
from essentia.standard import *
import essentia.standard as es
import pandas as pd
import scipy
import sklearn
# librosa
import librosa
import librosa.display
import os
from os import listdir
from os.path import isfile, join, basename, splitext

def create_chroma_df(chroma):
  chroma_mean = np.mean(chroma, axis = 1)
  chroma_std = np.std(chroma, axis = 1)

  chroma_df = pd.DataFrame()
  for i in range(0,12):
    chroma_df['chroma ' + str(i) + ' mean'] = chroma_mean[i]
    chroma_df['chroma ' + str(i) + ' std'] = chroma_std[i]
  chroma_df.loc[0] = np.concatenate((chroma_mean, chroma_std), axis = 0)

  return chroma_df

def create_mfccs_df(mfccs):
	mfccs_mean = np.mean(mfccs, axis = 1)
	mfccs_std = np.std(mfccs, axis = 1)
   
	mfccs_df = pd.DataFrame()
	for i in range(0,13):
		mfccs_df['mfccs ' + str(i) + ' mean'] = mfccs_mean[i]
		mfccs_df['mfccs ' + str(i) + ' std'] = mfccs_std[i]
	mfccs_df.loc[0] = np.concatenate((mfccs_mean, mfccs_std), axis = 0)
	
	return mfccs_df

def create_rms_df(rms):
	rms_mean = np.mean(rms)
	rms_std = np.std(rms)
	rms_skew = scipy.stats.skew(rms, axis = 1)[0]
	
	rms_df = pd.DataFrame()
	rms_df['rms mean'] = rms_mean
	rms_df['rms std'] = rms_std
	rms_df['rms skew'] = rms_skew
	rms_df.loc[0] = np.concatenate((rms_mean, rms_std, rms_skew), axis = 0)
	
	return rms_df

def create_spectral_df(cent, contrast, rolloff, flatness):
	
	# spectral centroids values
	cent_mean = np.mean(cent)
	cent_std = np.std(cent)
	cent_skew = scipy.stats.skew(cent, axis = 1)[0]

	# spectral contrasts values
	contrast_mean = np.mean(contrast, axis = 1)
	contrast_std = np.std(contrast, axis = 1)
	
	# spectral rolloff points values
	rolloff_mean = np.mean(rolloff)
	rolloff_std = np.std(rolloff)
	rolloff_skew = scipy.stats.skew(rolloff, axis = 1)[0]
	
	# spectral flatness values
	flat_mean = np.mean(flatness)
	flat_std = np.std(flatness)
	flat_skew = scipy.stats.skew(flatness, axis = 1)[0]

	spectral_df = pd.DataFrame()
	collist = ['cent mean','cent std','cent skew',
			   'flat mean', 'flat std', 'flat skew',
			   'rolloff mean', 'rolloff std', 'rolloff skew']
	for i in range(0,7):
		collist.append('contrast ' + str(i) + ' mean')
		collist.append('contrast ' + str(i) + ' std')
	
	for c in collist:
		spectral_df[c] = 0
	data = np.concatenate((
		[cent_mean, cent_std, cent_skew], 
		[flat_mean, flat_std, flat_skew],
		[rolloff_mean, rolloff_std, rolloff_skew], 
		contrast_mean, contrast_std),
		axis = 0)
	spectral_df.loc[0] = data
	
	return spectral_df

def create_zrate_df(zrate):
	zrate_mean = np.mean(zrate)
	zrate_std = np.std(zrate)
	zrate_skew = scipy.stats.skew(zrate, axis = 1)[0]

	zrate_df = pd.DataFrame()
	zrate_df['zrate mean'] = 0
	zrate_df['zrate std'] = 0
	zrate_df['zrate skew'] = 0
	zrate_df.loc[0]=[zrate_mean, zrate_std, zrate_skew]
	
	return zrate_df


def create_beat_df(tempo):
	beat_df = pd.DataFrame()
	beat_df['tempo'] = tempo
	beat_df.loc[0] = tempo
	return beat_df
  
def extract_features(audio):
	y, sr = librosa.load(audio, sr=44100)
	y_harmonic = y
	#y_harmonic, y_percussive = librosa.effects.hpss(y)
	
	tempo, beat_frames = librosa.beat.beat_track(y=y_harmonic, sr=sr)
	beat_times = librosa.frames_to_time(beat_frames, sr=sr)
	beat_time_diff = np.ediff1d(beat_times)
	beat_nums = np.arange(1, np.size(beat_times))
	
	chroma = librosa.feature.chroma_cens(y=y_harmonic, sr=sr)
	
	mfccs = librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=13)
	
	cent = librosa.feature.spectral_centroid(y=y, sr=sr)
	
	flatness = librosa.feature.spectral_flatness(y=y)
	
	contrast = librosa.feature.spectral_contrast(y=y_harmonic,sr=sr)
	
	rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
	
	zrate = librosa.feature.zero_crossing_rate(y_harmonic)
	
	chroma_df = create_chroma_df(chroma)
	
	mfccs_df = create_mfccs_df(mfccs)
	
	spectral_df = create_spectral_df(cent, contrast, rolloff, flatness)
	
	zrate_df = create_zrate_df(zrate)
	
	beat_df = create_beat_df(tempo)
	
	final_df = pd.concat((chroma_df, mfccs_df, spectral_df, zrate_df, beat_df), axis = 1)
	
	return final_df