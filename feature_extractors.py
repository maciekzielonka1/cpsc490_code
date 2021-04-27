import numpy as np
import os
import librosa
from resemblyzer import normalize_volume

def extract_mfccs_mean(y, sr = 22050, n_mfccs = 13):
	mfccs = librosa.feature.mfcc(y, n_mfcc=13)
	mfccs_mean = np.mean(mfccs, axis = 1)
	return mfccs_mean

def extract_f0(y):
	f_0 = librosa.pyin(y, librosa.note_to_hz('C2'), librosa.note_to_hz('C7'))[0]
	mean_f_0 = np.nanmean(f_0)
	if np.isnan(mean_f_0):
		mean_f_0 = 0.0
	return np.array([mean_f_0])

def extract_rms(y):
	rms = librosa.feature.rms(y)
	return np.array([rms])

def extract_zcr(y):
	zcr = librosa.feature.zero_crossing_rate(y)
	mean_zcr = np.mean(zcr)
	return np.array([mean_zcr])

def extract_centroid(y):
	centroid = librosa.feature.spectral_centroid(y)
	mean_centroid = np.mean(centroid)
	return np.array([mean_centroid])

def extract_mel_spectrogram(y):
	mel = librosa.feature.melspectrogram(y)
	mean_mel = np.mean(mel)
	return np.array([mean_mel])

def extract_chroma(y):
	chroma = librosa.feature.chroma_cens(y)
	mean_chroma = np.mean(chroma, axis = 1)
	return mean_chroma

def extract_features(wav_chunk):
	"""
	wav_chunk: a numpy array where all the entries correspond to the waveform values
			of an original ".wav" file
	return - features: a set of features extracted from the wav_chunk
	"""
	wav_chunk = librosa.resample(wav_chunk, 22050, 16000)
	wav_chunk = normalize_volume(wav_chunk, -30)
	
	if wav_chunk.shape[0] == 0:
		return (np.zeros((1, 15)), label)
	mfccs_mean = extract_mfccs_mean(wav_chunk)
	f_0 = extract_f0(wav_chunk)
	rms = extract_rms(wav_chunk)
	zcr = extract_zcr(wav_chunk)
	# centroid = extract_centroid(wav_chunk)
	# mel = extract_mel_spectrogram(wav_chunk)
	# chroma = extract_chroma(wav_chunk)
	features = np.concatenate((mfccs_mean, f_0, rms, zcr))
	return features

def extract_features_and_label(csv_file):
	"""
	csv_file: a ".csv" file containing a numpy array with waveform values for the specified chunk,
			as well as the last entry, which is a binary value marking the engagement/disengagement for that chunk
	returns - 
	features_array: a numpy array of extracted features for the specified chunk
	label: 0 for disengaged, 1 for engaged
	"""
	whole_array = np.loadtxt(csv_file, delimiter=",")
	wav_chunk = whole_array[:-1]
	label = whole_array[-1]
	features_array = extract_features(wav_chunk)
	features_array = np.reshape(features_array, (1, features_array.shape[0]))
	return (features_array, label)

def create_features_and_label_arrays(csv_dir):
	"""
	csv_dir: the directory of ".csv" files to loop through and extract features, which will then be used in the classifier
	returns - 
	features: a numpy array concatenating all of the extracted features
	labels: a numpy array concatenating all of the labels for the chunks
	"""
	first_file = True
	features = None
	labels = None
	for file in os.listdir(csv_dir):
		print("extracting features for", file)
		file_path = os.path.join(csv_dir, file)
		result = extract_features_and_label(file_path)
		if first_file:
			features = result[0]
			labels = result[1]
			first_file = False
		else:
			features = np.append(features, result[0], axis=0)
			labels = np.append(labels, result[1])
	return features, labels