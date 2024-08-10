import os
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.signal import butter, spectrogram, lfilter, windows
from IPython.display import Audio

def list_wav(wav_dir):
    filenames = os.listdir(wav_dir)
    wavs = []
    for f in filenames:
        if f.endswith(".WAV") or f.endswith(".wav"):
            wavs.append(f)
    return wavs

def parse_filename(filename):
    ind, date = filename.split("_")[:2]
    year = date[:4]
    month = date[4:6]
    day = date[6:]
    return ind, int(year), int(month), int(day)

# find songtype from filename
def parse_songtype(filename, pattern=r'(\d{3})([a-zA-Z])'):
    match = re.findall(pattern, filename.split(".")[0])
    if (len(match) == 1):
        return "".join(match[0])
    else:
        print(f"Cannot parse songtype from {filename}")

def plot_spec(aud, fs, NFFT=256, cmap="binary", save_path=None, 
              max_freq=None):
    # compute how much padding needed
    pad_to = NFFT-len(aud)%NFFT 
    s = plt.specgram(aud, Fs=fs, pad_to=pad_to, cmap=cmap)

    if (max_freq is not None):
        plt.ylim([0,max_freq])

    if (save_path is not None):
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.savefig(save_path, dpi=200)
        plt.close()
    
    return s

# ref: https://www.kaggle.com/code/mauriciofigueiredo/methods-for-sound-noise-reduction
def filter(y, sr, thresh=2000, t="highpass"):
    b, a = butter(10, thresh/(sr/2), btype=t)
    yf = lfilter(b, a, y)
    return yf

# scale float array to int16 array (song, post filtering)
# important for saving with scipy.wavfile
# see: 
def scale_song(song, max_int16=2**15-1):
    return np.int16(song / np.max(np.abs(song)) * max_int16)

# play audio segment given song, fs, time step (t0, t1)
def play_segment(song, ts, fs=44100):
    t0, t1 = ts
    return Audio(song[t0:t1], rate=fs)

# generate a frequency filter centered
def generate_exp_win(c, nbins, width=5, tau=1):
    bg = np.zeros(nbins)

    window = windows.exponential(M=width, tau=1)
    pre_c = np.floor(width/2).astype(int)
    post_c = np.ceil(width/2).astype(int)

    bg[c-pre_c : c+post_c] = window

    return bg

def normalize(series):
    max_v, min_v = max(series), min(series)
    return (series-min_v) / (max_v - min_v)