# Modified from https://medium.com/saarthi-ai/who-spoke-when-build-your-own-speaker-diarization-module-from-scratch-e7d725ee279

from resemblyzer import  preprocess_wav, VoiceEncoder, hparams
from pathlib import Path
from spectralcluster import SpectralClusterer
import librosa

def generate_embeddings(audio):
  """
  Creates embeddings using Resemblyzer's encoder. https://github.com/resemble-ai/Resemblyzer
  audio - a file path to a .wav file
  """
  wav_fpath = Path(audio)
  wav = preprocess_wav(wav_fpath)
  encoder = VoiceEncoder("cpu")
  _, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True, rate=16)
  print(cont_embeds.shape)
  return cont_embeds, wav_splits

def spectral_cluster_predict(cont_embeds):
  """
  Perform a spectral clustering on the embeddings
  cont_embeds - embeddings of preprocessed audio files
  """
  clusterer = SpectralClusterer(
      min_clusters=2,
      max_clusters=100,
      p_percentile=0.90,
      gaussian_blur_sigma=1)

  labels = clusterer.predict(cont_embeds)
  return labels

def create_labelling(labels, wav_splits):
  """
  Create a list of labels corresponding to labels and time segments in the format:
  ('label', start_time, end_time)
  """
  times = [((s.start + s.stop) / 2) / hparams.sampling_rate for s in wav_splits]
  labelling = []
  start_time = 0

  for i,time in enumerate(times):
      if i>0 and labels[i]!=labels[i-1]:
          temp = [str(labels[i-1]),start_time,time]
          labelling.append(tuple(temp))
          start_time = time
      if i==len(times)-1:
          temp = [str(labels[i]),start_time,time]
          labelling.append(tuple(temp))

  return labelling

def wav_file_diarization(audio):
  """
  Diarizes an audio file, labelling which speaker is speaking at specific points in the clip
  """
  cont_embeds, wav_splits = generate_embeddings(audio)
  labels = spectral_cluster_predict(cont_embeds)
  labelling = create_labelling(labels, wav_splits)
  return labelling