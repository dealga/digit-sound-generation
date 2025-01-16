import librosa
import numpy as np
import os
import pickle

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
    ##responsible to save the features and the min max values

    def __init__(self, feature_save_dir, min_max_values_save_dir):
        self.feature_save_dir = feature_save_dir
        self.min_max_values_save_dir = min_max_values_save_dir
    
    def save_feature(self, feature, file_path):
        save_path = self._generate_save_path(file_path)

        np.save(save_path, feature)

        return save_path
    
    def _generate_save_path(self, file_path):
        file_name = os.path.split(file_path)[1]  ##eg file_path = /home/user/audio.wav 
                                                 ##os.path.split(file_path) = ('/home/user', 'audio.wav')
                                                 ##os.path.split(file_path)[1] = 'audio.wav'
        save_path = os.path.join(self.feature_save_dir, file_name + ".npy")

        return save_path
    
    def save_min_max_values(self, min_max_values):
        save_path = os.path.join(self.min_max_values_save_dir, "min_max_values.pkl")

        self._save(min_max_values, save_path)

    @staticmethod
    def _save(data, save_path):
        with open(save_path, "wb") as f:
            pickle.dump(data, f)


class PreprocessingPipeline:
    ##preprocessing pipeline processes audio files in a directory, applying 
    ##the following steps each file:
    ##1. Load the audio file
    ##2. Pad the audio file
    ##3. Extract the log spectrogram
    ##4. Normalise the log spectrogram
    ##5. Save the log spectrogram
        
    def __init__(self):
        self.loader = None
        self.padder = None
        self.extractor = None                ##we dont name this log_spectrogram_extractor 
                                             ##as extractor could be anytype of extractor
                                             ##in this particular case we are using log_spectrogram_extractor
                                             ##but we could also add any type of extractor
                                             ##we could name the class log_spectrogram_extractor
                                             ##to extractor and implement log_spectrogram_extractor
                                             ##as a method in that class and also implement 
                                             ##other types of extractors as methods in that class
                                             ##like mfcc_extractor, chroma_extractor etc

        self.normaliser = None

        self.saver = None

        self.min_max_values = {}
        
    
    def process(self, audio_files_dir):
        for root, _, files in os.walk(audio_files_dir):
            for file in files:
                file_path = os.path.join(root, file)
                self._process_file(file_path)
                print(f"Processed file {file}")

        self.saver.save_min_max_values(self.min_max_values)


    def _process_file(self, file_path):
        ##this method goes through all the 5 steps of the preprocessing pipeline

        signal = self.loader.load(file_path)

        if self._is_padding_necessary(signal):
            signal = self._apply_padding(signal)
        
        feature = self.extractor.extract(signal)
        norm_feature = self.normaliser.normalise(feature)
        save_path = self.saver.save_feature(norm_feature, file_path)
        self._store_min_max(save_path, feature.min(), feature.max())

    
    def _is_padding_necessary(self, signal):
        num_expected_samples = int(self.loader.sample_rate * self.loader.duration)

        if len(signal) < num_expected_samples:
            return True

        return False

    def _apply_padding(self, signal):
        num_missing_samples = int(self.loader.sample_rate * self.loader.duration) - len(signal)

        padded_signal = self.padder.right_pad(signal, num_missing_samples)

        return padded_signal

    def _store_min_max(self, save_path, min_val, max_val):
        self.min_max_values[save_path] = {      ##we create a new key in the min_max_values dictionary
            "min": min_val,                     ##the key is the save_path
            "max": max_val
        }

if __name__ == "__main__":
    FRAME_SIZE = 512
    HOP_LENGTH = 256
    DURATION = 0.74
    SAMPLE_RATE = 22050
    MONO = True

    SPECTROGRAMS_SAVE_DIR = r"C:\Users\user\Desktop\digit sound using vae\digit-sound-generation\spectrograms"
    os.makedirs(SPECTROGRAMS_SAVE_DIR, exist_ok=True)

    MIN_MAX_VALUES_SAVE_DIR = r"C:\Users\user\Desktop\digit sound using vae\digit-sound-generation\min_max_values"
    os.makedirs(MIN_MAX_VALUES_SAVE_DIR, exist_ok=True)

    FILES_DIR = r"C:\Users\user\Desktop\digit sound using vae\digit-sound-generation\audio\recordings"


    loader = loader(SAMPLE_RATE, DURATION, MONO)
    padder = Padder()  ##takes in default value of mono i.e. set to constant
    log_spectrogram_extractor = LogSpectrogramExtractor(FRAME_SIZE, HOP_LENGTH)
    min_max_normaliser = MinMaxNormaliser(0, 1)

    saver = Saver(SPECTROGRAMS_SAVE_DIR, MIN_MAX_VALUES_SAVE_DIR)

    preprocessing_pipeline = PreprocessingPipeline()
    preprocessing_pipeline.loader = loader
    preprocessing_pipeline.padder = padder
    preprocessing_pipeline.extractor = log_spectrogram_extractor
    preprocessing_pipeline.normaliser = min_max_normaliser
    preprocessing_pipeline.saver = saver
    

    preprocessing_pipeline.process(FILES_DIR)