# library
import librosa
import librosa.display
import pylab
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1. Get the file path
"""P1_HHc = "Personal/Participant_1/P1_HHclosed_Personal.wav"
P1_HHo = "Personal/Participant_1/P1_HHopened_Personal.wav"
P1_Kd = "Personal/Participant_1/P1_Kick_Personal.wav"
P1_Sd = "Personal/Participant_1/P1_Snare_Personal.wav"""""
# 1.1 Get wav file
P1_impro_wav = "Personal/Participant_1/P1_Improvisation_Personal.wav"



def spectogram(wav_file):

    # Load the audio as a waveform `y`, store the sampling rate as `sr`
    y, sr = librosa.load(wav_file, sr=None, mono=True, dtype=np.float32)

    # Plot spectrogram
    spectrogram = pylab.specgram(y, Fs=sr)
    print("\n", spectrogram[0].shape, spectrogram[1].shape)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()
    plt.savefig('P1_impro_spectrogram.png')

def mel_spectogram(wav_file):
    # Load the audio as a waveform `y`, store the sampling rate as `sr`
    y, sr = librosa.load(wav_file, sr=None, mono=True, dtype=np.float32)
    plt.plot(y)
    plt.show()
    plt.clf()

    # Plot mel_spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=200, win_length=2000)
    print("\n", mel_spectrogram.shape)
    librosa.display.specshow(librosa.power_to_db(mel_spectrogram, ref=np.max), y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.savefig('P1_impro_mel_spectrogram.png')
    plt.show()


#spectogram(wav_file=P1_impro_wav)
#mel_spectogram(wav_file=P1_impro_wav)

