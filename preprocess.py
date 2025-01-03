import librosa
import numpy as np


class loader:
    def __init__(self, sample_rate, duration, mono):
        self.sample_rate = sample_rate
        self.duration = duration
        self.mono = mono

    def load(self, file_path):
        #librosa.load returns a tuple containing the audio signal and the sample rate so we use [0]
        signal = librosa.load(file_path, sr=self.sample_rate,
                                         duration=self.duration, 
                                         mono=self.mono)[0]

        return signal

class Padder:
    def __init__(self, mode="constant"):##mode = constant means that it will pad with a constant value,
                                        ## 0 by default
        self.mode = mode                 ##like a thin wrapper around numpy 

    def left_pad(self, array, num_missing_items):
        ##eg array [1,2,3] num_missing_items = 2 => [0,0,1,2,3]   i.e. we are left zero padding 

        padded_array = np.pad(array, (num_missing_items, 0), mode=self.mode) ##in (num_missing_items, 0) 
                                                                             ##num_missing_items is the number 
                                                                             ##of zeros to pad with
                                                                             ##0 number of zeros to pad 
                                                                             #with in the right
        return padded_array

    def right_pad(self, array, num_missing_items):
        ##eg array [1,2,3] num_missing_items = 2 => [1,2,3,0,0]   i.e. we are right padding

        padded_array = np.pad(array, (0, num_missing_items), mode=self.mode)
        return padded_array
        


    

class LogSpectrogramExtractor:
    ##This class is used to extract the log spectrogram (in dB) 
    ##from a time-series signal
    def __init__(self, frame_size, hop_length):
        self.frame_size = frame_size
        self.hop_length = hop_length

    def extract(self, signal): 

        ##stft is short time fourier transform

        stft = librosa.stft(signal, n_fft=self.frame_size, hop_length=self.hop_length)[:-1]
        ##we do [:-1] because the last frame is usually not complete
        ##i.e. we get of the shape (1+frame_size/2, num_frames)
        ##we dont need num_frames
        ##and for the first tuple we get the magnitude of the stft
        ##but it will become odd when we /2 and then add 1 for eg 1024/2 = 512 + 1 = 513
        ##so we dont need removig the last frame

        ##here, 513 is the number of frequency bins
        ##frequency bins are the number of frequencies we are considering
        ##in the stft

        spectrogram = np.abs(stft)
        log_spectrogram = librosa.amplitude_to_db(spectrogram)

        return log_spectrogram

class MinMaxNormaliser:
    """applies minmax normalisation to an array"""

    def __init__(self, min_val, max_val):
        self.min = min_val
        self.max = max_val

    def normalise(self, array):
        normalised_array = (array - array.min()) / (array.max() - array.min())


        ##eg array = [1,2,3] array.min() = 1 array.max() = 3
        ##normalised_array = (array - 1) / (3 - 1) = [0, 0.5, 1]

        normalised_array = normalised_array * (self.max - self.min) + self.min 
        ##self.max and self.min are the values we want to normalise between
        ##eg self.min = 0 self.max = 1
        ##normalised_array = normalised_array * (1 - 0) + 0 = [0, 0.5, 1]


        return normalised_array

    def denormalise(self, array, original_min, original_max):
        denormalised_array = (array - self.min) / (self.max - self.min)
        denormalised_array = denormalised_array * (original_max - original_min) + original_min

        return denormalised_array



class Saver:
    pass

class PreprocessingPipeline:
    ##preprocessing pipeline processes audio files in a directory, applying 
    ##the following steps each file:
    ##1. Load the audio file
    ##2. Pad the audio file
    ##3. Extract the log spectrogram
    ##4. Normalise the log spectrogram
    ##5. Save the log spectrogram
        
    def __init__(self, steps):
        self.steps = steps