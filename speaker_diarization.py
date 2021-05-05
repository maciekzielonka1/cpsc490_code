from resemblyzer import  preprocess_wav, VoiceEncoder, hparams
from sklearn.cluster import SpectralClustering
import librosa
import numpy as np
import matplotlib.pyplot as plt
import math
from umap.umap_ import UMAP
from resemblyzer import VoiceEncoder, preprocess_wav
from helpers import convertSecs
from pydub import silence, AudioSegment
from resemblyzer import VoiceEncoder

def generate_embeddings(wav_array):
    """
    Creates embeddings using Resemblyzer's encoder. https://github.com/resemble-ai/Resemblyzer
    wav_array: a waveform array that needs to be encoded
    return - 
    cont_embeds: embeddings generated 
    wav_splits: a list containing information of where the partial embeddings start and where they stop
    """
    encoder = VoiceEncoder("cpu")
    _, cont_embeds, wav_splits = encoder.embed_utterance(wav_array, return_partials=True, rate=4)
    return cont_embeds, wav_splits

def spectral_cluster_predict(cont_embeds, n_clusters):
    """
    Perform a spectral clustering on the embeddings
    cont_embeds: embeddings of audio segments
    n_clusters: number of clusters to generate
    returns - 
    labels: labels for each embedding
    """
    clusterer = SpectralClustering(n_clusters=n_clusters, assign_labels="discretize", random_state=0)
    fitting = clusterer.fit(cont_embeds)
    labels = fitting.labels_
    return labels

def label_wav_splits(y, wav_splits, labels, sr = 22050):
    """
    Create a list marking when labels occur during the wav segment
    wav_splits: a list containing information of where the partial embeddings start and where they stop
    labels: labels for each wav_split
    sr: the sampling rate for the wav file
    returns - 
    diarization_dict: a dictionary specifying what each label corresponds to "Child", "Adult", "Silence"
    labelling: a list of ("Number label", "Label start time", "Label end time", "String Label")
    """
    times = [((s.start + s.stop) / 2) / sr for s in wav_splits]
    labelling = []
    start_time = 0
    diarization_dict = {}
    adult_set = False
    silence_set = False
    rms = np.mean(librosa.feature.rms(y))

    for i,split in enumerate(wav_splits):
        start = split.start
        end = split.stop
        mid = (start + end)//2
        if i>0 and labels[i]!=labels[i-1]:
            lbl = labels[i-1]
            if lbl not in diarization_dict.keys():
                if not silence_set and np.mean(librosa.feature.rms(y[start_time:mid])) < rms*.5:
                    cls = "Silence"
                    diarization_dict[lbl] = cls
                    silence_set = True
                elif not adult_set:
                    cls = "Adult"
                    adult_set = True
                    diarization_dict[lbl] = cls
                else:
                    cls = "Child"
                    diarization_dict[lbl] = cls
            temp = [str(labels[i-1]),start_time,mid, diarization_dict[lbl]]
            labelling.append(tuple(temp))
            start_time = mid
        if i==len(times)-1:
            lbl = labels[i-1]
            temp = [str(labels[i]),start_time,mid, diarization_dict[lbl]]
            labelling.append(tuple(temp))
    return diarization_dict, labelling

def show_plots(cont_embeds, labels):
    plt.imshow(cont_embeds@cont_embeds.T)
    projections = create_projections(cont_embeds)
    plot_embeddings(projections, labels)

def create_projections(embeds):
    reducer = UMAP()
    projs = reducer.fit_transform(embeds)
    return projs

def plot_embeddings(projs, labels):
    _, ax = plt.subplots(figsize=(6, 6))
    colors = ['b', 'r', 'g']
    print(np.unique(labels))
    for i, label in enumerate(np.unique(labels)):
        speaker_projs = projs[label == labels]
        marker = "o"
        ax.scatter(*speaker_projs.T, marker=marker, label=label, color=colors[i])
    plt.show()

def diarize(y, n_clusters):
    embeddings, wav_splits = generate_embeddings(y)
    labels = spectral_cluster_predict(embeddings, n_clusters)
    diarization_dict, labelling = label_wav_splits(y, wav_splits, labels)
    return labelling

def print_diarization(labelling, sr=22050):
    for label in labelling:
        start = int(label[1])/sr
        end = int(label[2])/sr
        lbl = label[3]
        print(convertSecs(start), "-", convertSecs(end), "--", lbl)

def diarize_with_pydub(wav_file, min_silence_len=500, thresh=-40):
    y, sr = librosa.load(wav_file)
    audio_segment = AudioSegment.from_file(wav_file, frame_rate=sr)
    silent_segments = silence.detect_silence(audio_segment, min_silence_len, thresh)
    silent_segments_in_frames = [(int((s[0] / 1000) * sr), int((s[1] / 1000) * sr)) for s in silent_segments]
    last_frame = len(y)
    all_segments, segments_for_embedding = create_all_segments(silent_segments_in_frames, last_frame)
    embeddings = embed_segments(y, segments_for_embedding)
    labels = spectral_cluster_predict(embeddings, 2)
    show_plots(embeddings, labels)
    silence_first = silent_segments_in_frames[0][0] == 0
    labelling = create_labellings_with_pydub(y, all_segments, labels, silence_first)
    return labelling, silent_segments_in_frames
  
def create_labellings_with_pydub(y, segments, labels, silence_first):
    adult_set = False
    adult_label = labels[0]
    label_idx = 0
    labelling = []
    for i, segment in enumerate(segments):
        start = segment[0]
        end = segment[1]
        if silence_first:
            if i % 2 == 0:
                labelling.append(('2', start, end, "Silence"))
            else:
                label = labels[label_idx]
                if label == adult_label:
                    speaker = "Adult"
                else:
                    speaker = "Child"
                labelling.append((label, start, end, speaker))
                label_idx += 1
        else:
            if i % 2 == 1:
                labelling.append(('2', start, end, "Silence"))
            else:
                label = labels[label_idx]
                if label == adult_label:
                    speaker = "Adult"
                else:
                    speaker = "Child"
                labelling.append((label, start, end, speaker))
                label_idx += 1
    return labelling

def embed_segments(y, segments_for_embedding):
    embeddings = []
    encoder = VoiceEncoder("cpu")
    for segment in segments_for_embedding:
        start = segment[0]
        end = segment[1]
        embedding = encoder.embed_utterance(y[start:end])
        embeddings.append(embedding)
    return np.array(embeddings)

def create_all_segments(silent_segments_in_frames, last_frame):
    all_segments = []
    segments_for_embedding = []
    chunk_start = 0
    for segment in silent_segments_in_frames:
        segment_start = segment[0]
        segment_end = segment[1]
        if chunk_start != segment_start:
            all_segments.append((chunk_start, segment_start))
            segments_for_embedding.append((chunk_start, segment_start))
        all_segments.append(segment)
        chunk_start = segment_end
    if chunk_start != last_frame:
        all_segments.append((chunk_start, last_frame))
        segments_for_embedding.append((chunk_start, last_frame))
    return all_segments, segments_for_embedding
