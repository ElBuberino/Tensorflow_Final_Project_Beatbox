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
"""
    # create color mapping for each category
    grey_dict = {
        "hhc": 0.2,     # high head closed
        "hho": 0.3,     # high head open
        "kd": 0.4,      # kick drum
        "sd": 0.5       # snare drum
    }

    # create blank target with the image-shape of our spectrogram
    target = np.zeros_like(spectrogram)
    # print onset times on target
    for i, val in enumerate(time_indexes):
        #target[:, val] = grey_dict[df_vals[i]]

        category = df_vals[i]

        color = grey_dict[category]

        target[:, val] = np.repeat(color, spectrogram.shape[0], axis=0)

    # append targets to label_list
    label_list.append(target)
"""
# create color maps for each category
hhc_cmap = ListedColormap(['white', 'black', 'grey'], name='hhc')
hho_cmap = ListedColormap(['white', 'red', 'grey'], name='hho')
kd_cmap = ListedColormap(['white', 'blue', 'grey'], name='kd')
sd_cmap = ListedColormap(['white', 'green', 'grey'], name='sd')

# create a dictionary to store the color maps for each category
cmap_dict = {
    "hhc": hhc_cmap,
    "hho": hho_cmap,
    "kd": kd_cmap,
    "sd": sd_cmap
}

# create target with the image-shape of our spectrogram
target = np.zeros_like(spectrogram)

# iterate through each onset time and corresponding category value
for i, val in enumerate(time_indexes):
    category = df_vals[i]

    # get the color map for the category and set the color of the target
    cmap = cmap_dict[category]
    color = cmap(0.5)

    # resize the color array to match the shape of the target column
    color = np.reshape(color, (spectrogram.shape[0]))

    target[:, val] = color

# append targets to label_list
label_list.append(target)



## Plot spectrograms and targets as png-images ##
# create an empty dictionary to store the paths of saved plots
plot_dict = {}

# set the figure size and dpi for all plots
fig = plt.figure(figsize=(8, 8), dpi=100)

# iterate through each pair of spectrograms and their corresponding labels
for idx, (spec, label) in enumerate(zip(spec_list, label_list)):
    # create a visual representation of the spectrogram using librosa and matplotlib
    spec_ex = librosa.display.specshow(librosa.power_to_db(spec, ref=np.max))
    # define a file path to save the spectrogram image using the current index
    filename_spec = f"test_plots/mel_spectrogram_{idx}.png"
    # save the current figure using the specified file path and clear the figure
    plt.savefig(filename_spec)
    plt.clf()
    # add the file path of the saved spectrogram to the dictionary using the current index
    plot_dict[f"spec_{idx}"] = os.path.abspath(filename_spec)

    # create a visual representation of the label using librosa and matplotlib
    label_ex = librosa.display.specshow(label)
    # define a file path to save the label image using the current index
    filename_label = f"test_plots/label_{idx}.png"
    # save the current figure using the specified file path and clear the figure
    plt.savefig(filename_label)
    plt.clf()
    # add the file path of the saved label to the dictionary using the current index
    plot_dict[f"label_{idx}"] = os.path.abspath(filename_label)


