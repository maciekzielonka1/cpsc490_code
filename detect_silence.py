# Modified Pydub's silence module (https://github.com/jiaaro/pydub/tree/master/pydub)
# to allow for more flexible silence detection
from pydub import AudioSegment, silence
from scipy.io.wavfile import read as wav_read
from statistics import mean, stdev

def db_to_float(db, using_amplitude=True):
    """
    Converts the input db to a float, which represents the equivalent
    ratio in power.
    """
    db = float(db)
    if using_amplitude:
        return 10 ** (db / 20)
    else:  # using power
        return 10 ** (db / 10)

def detect_silence(audio_segment, min_silence_len=1000, silence_thresh=-50, seek_step=1):
    """
    Returns a list of all silent sections [start, end] in milliseconds of audio_segment.
    Inverse of detect_nonsilent()
    audio_segment - the segment to find silence in
    min_silence_len - the minimum length for any silent section
    silence_thresh - the upper bound for how quiet is silent in dFBS
    seek_step - step size for interating over the segment in ms
    
    Instead of only using an absolute dBFS value as a threshold for detecting silence, 
    which is what pyDub's implementation does, this implementation also detects segments
    whose RMS values are less than the mean RMS of the entire segment by more than two
    standard deviations. This helps with filtering out noise that might be louder than the provided
    silence threshold.  
    """
    seg_len = len(audio_segment)

    whole_segment_rms = audio_segment.rms

    # you can't have a silent portion of a sound that is longer than the sound
    if seg_len < min_silence_len:
        return []

    # convert silence threshold to a float value (so we can compare it to rms)
    dbfs_silence_thresh = db_to_float(silence_thresh) * audio_segment.max_possible_amplitude

    # find silence and add start and end indicies to the to_cut list
    silence_starts = []

    # check successive (1 sec by default) chunk of sound for silence
    # try a chunk at every "seek step" (or every chunk for a seek step == 1)
    last_slice_start = seg_len - min_silence_len
    slice_starts = range(0, last_slice_start + 1, seek_step)

    # guarantee last_slice_start is included in the range
    # to make sure the last portion of the audio is searched
    if last_slice_start % seek_step:
        slice_starts = itertools.chain(slice_starts, [last_slice_start])
    
    # Loop through each segment once, recording the rms values, and record the 
    # standard deviation
    rms_of_each_segment = []
    for i in slice_starts:
        audio_slice = audio_segment[i:i + min_silence_len]
        rms_of_each_segment.append(audio_slice.rms)
    rms_standard_deviation = stdev(rms_of_each_segment)

    for i in slice_starts:
        audio_slice = audio_segment[i:i + min_silence_len]
        if audio_slice.rms <= dbfs_silence_thresh or audio_slice.rms <= (whole_segment_rms - 2*rms_standard_deviation):
            silence_starts.append(i)

    # short circuit when there is no silence
    if not silence_starts:
        return []

    # combine the silence we detected into ranges (start ms - end ms)
    silent_ranges = []

    prev_i = silence_starts.pop(0)
    current_range_start = prev_i

    for silence_start_i in silence_starts:
        continuous = (silence_start_i == prev_i + seek_step)

        # sometimes two small blips are enough for one particular slice to be
        # non-silent, despite the silence all running together. Just combine
        # the two overlapping silent ranges.
        silence_has_gap = silence_start_i > (prev_i + min_silence_len)

        if not continuous and silence_has_gap:
            silent_ranges.append([current_range_start,
                                  prev_i + min_silence_len])
            current_range_start = silence_start_i
        prev_i = silence_start_i

    silent_ranges.append([current_range_start,
                          prev_i + min_silence_len])

    return silent_ranges

def get_intervals_of_silence(audio, silence_thresh=-50):
  silence_intervals = detect_silence(audio, min_silence_len=1000, silence_thresh=silence_thresh)
  silence_intervals = [((start/1000),(stop/1000)) for start,stop in silence_intervals] #convert to sec
  return silence_intervals

def get_silence_info(audio, silence_thresh=-50):
  """ 
  Returns a dict with relevant information regarding the silence of an audio clip.
  audio - A path to a .wav file
  silence_thresh - A user-specified threshold (in dBFS) under which segments are detected as silence
  """
  audio = AudioSegment.from_wav(audio)
  silence_intervals = get_intervals_of_silence(audio, silence_thresh)
  silence_info = {}
  silence_info['intervals'] = silence_intervals
  silence_info['durations'] = [y - x for x, y in silence_intervals]
  silence_info['max_silence'] = max(silence_info['durations'])
  silence_info['min_silence'] = min(silence_info['durations'])
  silence_info['avg_silence'] = mean(silence_info['durations'])
  silence_info['frequency'] = len(silence_info['durations'])
  silence_info['total_silence'] = sum(silence_info['durations'])
  silence_info['silence_ratio'] = silence_info['total_silence']/(len(audio)/1000)
  return silence_info