import pandas as pd

def extract_features_from_diarized_interview(y, diarization_labels):
    n_chunks = len(diarization_labels)
    num_silent_chunks = 0
    num_child_chunks = 0
    total_child_len = 0
    total_silence_len = 0
    num_adult_chunks = 0
    total_adult_len = 0
    num_engaged_chunks = 0
    num_disengaged_chunks = 0
    duration_engaged = 0
    duration_disengaged = 0
    for label in diarization_labels:
        chunk_start = label[1]
        chunk_end = label[2]
        speaker = label[3]
        chunk_len = chunk_end - chunk_start
        if speaker == 'Silence':
            num_silent_chunks += 1
            total_silence_len += chunk_len
        if speaker == 'Adult':
            num_adult_chunks += 1
            total_adult_len += chunk_len
        if speaker == 'Child':
            num_child_chunks += 1
            total_child_len += chunk_len
            wav_chunk = y[chunk_start:chunk_end]
            features = extract_features(wav_chunk)
            label = tree_clf.predict([features])[0]
            if label == 1:
                num_engaged_chunks += 1
                duration_engaged += chunk_len
            else:
                num_disengaged_chunks += 1
                duration_disengaged += chunk_len
    data = {"n_chunks": n_chunks, 
    "num_silent_chunks": num_silent_chunks, 
    "total_silence_len": total_silence_len, 
    "num_child_chunks": num_child_chunks,
    "total_child_len": total_child_len,
    "num_engaged_chunks": num_engaged_chunks, 
    "duration_engaged": duration_engaged,
    "num_disengaged_chunks": num_disengaged_chunks,
    "duration_disengaged": duration_disengaged,
    "num_adult_chunks": num_adult_chunks,
    "total_adult_len": total_adult_len
    }
    return pd.DataFrame(data, index=[0])
