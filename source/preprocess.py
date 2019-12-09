import numpy as np 
import librosa

sample_rate = 16000
# Load speech signal and extract features from it
def load_and_process(audio_signal):
    audio_signal = audio_signal/np.amax(audio_signal)
    zcr_signal = librosa.feature.zero_crossing_rate(audio_signal, frame_length=512, hop_length=410) #(1,32)
    mfcc_signal = librosa.feature.mfcc(audio_signal,sample_rate, n_fft=1024, hop_length=410) # (20,32)
    centr_signal = librosa.feature.spectral_centroid(audio_signal, sample_rate, n_fft=1024, hop_length=410)# (1,32)
    rolloff_signal = librosa.feature.spectral_rolloff(audio_signal,n_fft=1024, hop_length=410) # (1,32)
    # get features matrix by concatenating all features
    features_mat = np.concatenate((zcr_signal, mfcc_signal, centr_signal, rolloff_signal), axis=0)
    return features_mat