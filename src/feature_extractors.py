import numpy as np
import pandas as pd
import os
import librosa
from resemblyzer import normalize_volume
import scipy

def extract_mfccs_df(y, sr = 22050, n_mfccs = 13):
	"""
	Extracts `n_mfccs` number of mel frequency coefficients. 
	"""
	mfccs = librosa.feature.mfcc(y, n_mfcc=13)
	mfccs_mean = np.mean(mfccs, axis = 1)
	mfccs_std = np.std(mfccs, axis=1)
	mfccs_df = pd.DataFrame()
	for i in range(0, 13):
		mfccs_df['mfccs ' + str(i) + ' mean'] = mfccs_mean[i]
		mfccs_df['mfccs ' + str(i) + ' std'] = mfccs_std[i]
	mfccs_df.loc[0] = np.concatenate((mfccs_mean, mfccs_std), axis=0)
	return mfccs_df

def extract_f0_df(y):
	"""
	Extracts the f0 value from the wav chunk
	"""
	f_0 = librosa.pyin(y, librosa.note_to_hz('C2'), librosa.note_to_hz('C7'))[0]
	mean_f_0 = np.nanmean(f_0)
	if np.isnan(mean_f_0):
		mean_f_0 = 0
	f0_df = pd.DataFrame()
	f0_df['f0'] = mean_f_0
	return f0_df

def extract_rms_df(y):
	"""
	Extracts the Root-Mean-Square value for each frame in y
	"""
	rms = librosa.feature.rms(y)
	rms_mean = np.mean(rms)
	rms_std = np.std(rms)
	rms_skew = scipy.stats.skew(rms, axis = 1)[0]

	rms_df = pd.DataFrame()
	rms_df['rms mean'] = rms_mean
	rms_df['rms std'] = rms_std
	rms_df['rms skew'] = rms_skew
	rms_df.loc[0] = np.array([rms_mean, rms_std, rms_skew])

	return rms_df

def extract_zcr_df(y):
	"""
	Extracts the zero crossing rate of y
	"""
	zrate = librosa.feature.zero_crossing_rate(y)
	zrate_mean = np.mean(zrate)
	zrate_std = np.std(zrate)
	zrate_skew = scipy.stats.skew(zrate, axis = 1)[0]

	zrate_df = pd.DataFrame()
	zrate_df['zrate mean'] = 0
	zrate_df['zrate std'] = 0
	zrate_df['zrate skew'] = 0
	zrate_df.loc[0]=[zrate_mean, zrate_std, zrate_skew]

	return zrate_df

def extract_chroma_df(y, sr):
	"""
	Extracts the Chroma Energy Normalized values of y
	"""
	chroma_cens = librosa.feature.chroma_cens(y, sr)
	chroma_cens_mean = np.mean(chroma_cens, axis = 1)
	chroma_cens_std = np.std(chroma_cens, axis = 1)

	chroma_stft = librosa.feature.chroma_stft(y, sr)
	chroma_stft_mean = np.mean(chroma_stft, axis = 1)
	chroma_stft_std = np.std(chroma_stft, axis = 1)

	chroma_cqt = librosa.feature.chroma_cqt(y, sr)
	chroma_cqt_mean = np.mean(chroma_cqt, axis = 1)
	chroma_cqt_std = np.std(chroma_cqt, axis = 1)

	chroma_df = pd.DataFrame()
	for i in range(0,12):
		chroma_df['chroma_cens ' + str(i) + ' mean'] = chroma_cens_mean[i]
		chroma_df['chroma_cens ' + str(i) + ' std'] = chroma_cens_std[i]
		chroma_df['chroma_stft ' + str(i) + 'mean'] = chroma_stft_mean[i]
		chroma_df['chroma_stft ' + str(i) + 'std'] = chroma_stft_std[i]
		chroma_df['chroma_cqt ' + str(i) + 'mean'] = chroma_cqt_mean[i]
		chroma_df['chroma_cqt ' + str(i) + 'std'] = chroma_cqt_std[i]
	
	chroma_df.loc[0] = np.concatenate((chroma_cens_mean, chroma_cens_std, chroma_stft_mean, chroma_stft_std, chroma_cqt_mean, chroma_cqt_std), axis = 0)

	return chroma_df

def extract_spectral_df(y, sr):
	cent = librosa.feature.spectral_centroid(y, sr=sr)
	flatness = librosa.feature.spectral_flatness(y)
	contrast = librosa.feature.spectral_contrast(y,sr=sr)
	rolloff = librosa.feature.spectral_rolloff(y, sr=sr)
	bandwidth = librosa.feature.spectral_bandwidth(y, sr=sr)
	mel = librosa.feature.melspectrogram(y)

	band_mean = np.mean(bandwidth)
	band_std = np.std(bandwidth)
	band_skew = scipy.stats.skew(bandwidth, axis=1)[0]

	# spectral centroids values
	cent_mean = np.mean(cent)
	cent_std = np.std(cent)
	cent_skew = scipy.stats.skew(cent, axis = 1)[0]

	# mel spectrogram values
	mel_mean = np.mean(mel)
	mel_std = np.std(mel)
	mel_skew = scipy.stats.skew(mel, axis=1)[0]

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
				'rolloff mean', 'rolloff std', 'rolloff skew',
				'mel mean', 'mel std', 'mel skew',
				'band mean', 'band std', 'band skew']

	for c in collist:
		spectral_df[c] = 0
	data = np.concatenate((
		[cent_mean, cent_std, cent_skew], 
		[flat_mean, flat_std, flat_skew],
		[rolloff_mean, rolloff_std, rolloff_skew],
		[mel_mean, mel_std, mel_skew],
		[band_mean, band_std, band_skew]),
		axis = 0)
	spectral_df.loc[0] = data

	return spectral_df

def extract_tonnetz_df(y, sr):
	tonnetz = librosa.feature.tonnetz(y, sr)
	tonnetz_mean = np.mean(tonnetz, axis=1)
	tonnetz_std = np.std(tonnetz, axis=1)
	tonnetz_df = pd.DataFrame()
	for i in range(6):
		tonnetz_df['tonnetz ' + str(i) + ' mean'] = tonnetz_mean[i]
		tonnetz_df['tonnetz ' + str(i) + ' std'] = tonnetz_std[i]
	tonnetz_df.loc[0] = np.concatenate((tonnetz_mean, tonnetz_std), axis=0)
	return tonnetz_df

def extract_features_from_chunk(wav_chunk):
	"""
	wav_chunk: a numpy array where all the entries correspond to the waveform values
		of an original ".wav" file
	return - features: a set of features extracted from the wav_chunk
	"""
	sr = 22050
	if wav_chunk.shape[0] == 0:
		return pd.DataFrame()
	mfccs_df = extract_mfccs_df(wav_chunk)
	rms_df = extract_rms_df(wav_chunk)
	zcr_df = extract_zcr_df(wav_chunk)
	f0_df = extract_f0_df(wav_chunk)
	spectral_df = extract_spectral_df(wav_chunk, sr)
	chroma_df = extract_chroma_df(wav_chunk, sr)
	tonnetz_df = extract_tonnetz_df(wav_chunk, sr)
	features_df = pd.concat((mfccs_df, rms_df, zcr_df, spectral_df, chroma_df, f0_df, tonnetz_df), axis=1)
	return features_df

def extract_features_from_csv(csv_file):
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
	features_df = extract_features_from_chunk(wav_chunk)
	if not features_df.empty:
		features_df['Engaged'] = label
	return features_df

def create_all_features_df(csv_dir):
	"""
	csv_dir: the directory of ".csv" files to loop through and extract features, which will then be used in the classifier
	returns - 
	features: a numpy array concatenating all of the extracted features
	labels: a numpy array concatenating all of the labels for the chunks
	"""
	all_features = []
	for file in os.listdir(csv_dir):
		print("extracting features for", file)
		file_path = os.path.join(csv_dir, file)
		features_df = extract_features_from_csv(file_path)
		if features_df.empty:
			continue
		all_features.append(features_df)
	return pd.concat(all_features)