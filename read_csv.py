import pandas as pd
import os
from glob import glob
from IPython.display import Audio
from scipy.io import wavfile
import librosa
import librosa.display
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import tensorflow as tf


# set path
PATH = "C:/Users/accou/OneDrive/Desktop/Arbeit/Studium/Master/Semester03/IANNwTF/final_project/Personal"
# set file ending
CSV = "*.csv"
WAV = "*.wav"
spec_list = []
label_list = []

# collect all csv files
all_csv_files = [file
                 for path, subdir, files in os.walk(PATH)
                 for file in glob(os.path.join(path, CSV))]

# collect all wav files
all_wav_files = [file
                 for path, subdir, files in os.walk(PATH)
                 for file in glob(os.path.join(path, WAV))]

for csv_file, wav_file in zip(all_csv_files, all_wav_files):

    ## Create inputs ##
    # Load the audio as a waveform `y`, store the sampling rate as `sr`
    y, sr = librosa.load(wav_file)
    # Create mel spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    # append spectrograms to spec_list
    spec_list.append(spectrogram)

    ## Create labels ##
    # read in csv file
    df = pd.read_csv(csv_file)
    # convert to numpy array
    df_arr = df.to_numpy()
    # create array with onsets
    df_onsets = df_arr[:, 0]
    # create array
    df_vals = df_arr[:, 1]
    # create list with time indices of onset times
    time_indexes = []
    # store onset times in the shape of our spectrogram
    times = librosa.times_like(spectrogram)
    # append each onset time into times_indexes list
    for i in df_onsets:
        time_indexes.append(np.abs(times - i).argmin())

    # create blank target with the image-shape of our spectrogram
    target = np.zeros_like(spectrogram)

    color_dict = {
    "hhc": 'b',
    "hho": 'g',
    "kd": 'c',
    "sd": 'w'
    }
    grey_dict = {
        "hhc": 50,
        "hho": 100,
        "kd": 150,
        "sd": 200
    }

    #target = np.zeros_like(spectrogram)
    for i, val in enumerate(time_indexes):
        plt.axvline(x=val,color=color_dict[df_vals[ind]])
        #target[:, val] = grey_dict[df_vals[i]]

    # save plot as variable
    plt_as_variable = plt.gcf()
    # append targets to label_list
    label_list.append(plt_as_variable) #target)

## Save lists as datasets ##
spec_ds = tf.data.Dataset.from_tensor_slices(spec_list)
label_ds = tf.data.Dataset.from_tensor_slices(label_list)