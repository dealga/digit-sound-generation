from tensorflow.keras.datasets import mnist
from variationalautoencodercopy import VAE
import os
import numpy as np

LEARNING_RATE = 0.0005
BATCH_SIZE = 64
EPOCHS = 150
SPECTROGRAMS_PATH = r"\Users\user\Desktop\digit sound using vae\digit-sound-generation\spectrograms"

def load_fsdd(spectrograms_path):
    x_train = []

    for root, _, file_names in os.walk(spectrograms_path):
        for file_name in file_names:
            file_path = os.path.join(root, file_name)
            spectrogram = np.load(file_path)     ##(n_bins, n_frames)  
            x_train.append(spectrogram)


    x_train = np.array(x_train)    ##we cast list to a numpy array

    ##CNNS expect 3D inputs BUT WE HAVE 2D inputs i.e. (n_bins, n_frames)

    ##we can add a channel dimension to the spectrogram

    x_train = np.expand_dims(x_train, -1)  ##(n_bins, n_frames, 1) here -1 means the last dimension 
                                           ##by default 1 is added at the end


    ##if we want to add a custom number of channels in the end, we can do it as 
           ## x_train = np.expand_dims(x_train, -1)  # Add single channel
           ## x_train = np.repeat(x_train, 3, axis=-1)  # Create 3 identical channels


    return x_train

def train(x_train, learning_rate, batch_size, epochs):
    vae = VAE(
        input_shape=(256, 64, 1),
        conv_filters=(512, 256, 128, 64, 32),
        conv_kernels=(3, 3, 3, 3, 3),
        conv_strides=(2, 2, 2, 2, (2, 1)),
        latent_space_dim=128
    )
    
    vae.compile(learning_rate=learning_rate)
    vae.fit(x_train, batch_size=batch_size, epochs=epochs)
    
    return vae

if __name__ == "__main__":
    x_train = load_fsdd(SPECTROGRAMS_PATH)
    vae = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    vae.save("model")