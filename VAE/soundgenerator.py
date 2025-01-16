from preprocess import MinMaxNormaliser

class SoundGenerator:
    ##this is responsible for generating the sound from spectrorams

    ##we use inverse short time fourier transform to generate the sound

    ##particularly we use the Griffin-Lim algorithm to generate the sound


    def __init__(self, vae, hop_length):
        self.vae = vae
        self.hop_length = hop_length
        self._min_max_normalizer = MinMaxNormaliser(0, 1)  ##we need to normalise the data 
                                                           ##before we feed it to the vae

    def generate(self, spectrograms, min_max_values):
        generated_spectrograms, latent_representations = self.vae.reconstruct(spectrograms)

        signals = self.convert_spectrograms_to_audio(generated_spectrograms, min_max_values)

        return signals, latent_representations

    def convert_spectrograms_to_audio(self, spectrograms, min_max_values):
        signals = []

        for spectrogram, min_max_values in zip(spectrograms, min_max_values):
            #reshape the spectrogram to the original shape because we 
            ##we added an extra dimension to the spectrogram for compatibility with the vae i.e. cnn
            log_spectrogram = spectrogram[ :, :, 0]   ##we are dropping the 3rd dimension
            ##apply denormalization
            denormalized_spectrogram = self._min_max_normalizer.denormalize(log_spectrogram, min_max_values["min"], min_max_values["max"])
            ##log spectrogram -> spectrogram 
            spectrogram = librosa.db_to_amplitude(denormalized_spectrogram) ##this is an inbuilt function in librosa

            ##we apply griffin lim algorithm to get the signal
            signal = librosa.griffinlim(spectrogram, hop_length=self.hop_length)

            signals.append(signal)

        return signals




        