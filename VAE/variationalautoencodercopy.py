from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose, Activation, Lambda, Layer
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.optimizers import Adam
import os
import pickle
import tensorflow as tf

class Sampling(Layer):
    def call(self, inputs):
        mu, log_variance = inputs
        epsilon = tf.random.normal(shape=tf.shape(mu))
        return mu + tf.exp(log_variance * 0.5) * epsilon

class VAE(Model):
    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim):
        super(VAE, self).__init__()
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim
        self.reconstruction_loss_weight = 1000
        
        self.encoder = None
        self.decoder = None
        self._num_conv_layers = len(conv_filters)
        self.shape_before_bottleneck = None
        self._model_input = None
        
        self._build()

    def _build(self):
        self._build_encoder()
        self._build_decoder()

    def _build_encoder(self):
        encoder_input = Input(shape=self.input_shape, name="encoder_input")
        x = encoder_input
        
        for i in range(self._num_conv_layers):
            x = Conv2D(
                filters=self.conv_filters[i],
                kernel_size=self.conv_kernels[i],
                strides=self.conv_strides[i],
                padding="same",
                name=f"encoder_conv_layer_{i+1}"
            )(x)
            x = ReLU(name=f"encoder_relu_{i+1}")(x)
            x = BatchNormalization(name=f"encoder_bn_{i+1}")(x)
        
        self.shape_before_bottleneck = K.int_shape(x)[1:]
        x = Flatten()(x)
        
        self.mu = Dense(self.latent_space_dim, name="mu")(x)
        self.log_variance = Dense(self.latent_space_dim, name="log_variance")(x)
        
        z = Sampling()([self.mu, self.log_variance])
        
        self.encoder = Model(encoder_input, [self.mu, self.log_variance, z], name="encoder")

    def _build_decoder(self):
        decoder_input = Input(shape=(self.latent_space_dim,), name="decoder_input")
        
        num_neurons = np.prod(self.shape_before_bottleneck)
        x = Dense(num_neurons, name="decoder_dense")(decoder_input)
        x = Reshape(self.shape_before_bottleneck)(x)
        
        for i in reversed(range(1, self._num_conv_layers)):
            x = Conv2DTranspose(
                filters=self.conv_filters[i],
                kernel_size=self.conv_kernels[i],
                strides=self.conv_strides[i],
                padding="same",
                name=f"decoder_conv_transpose_layer_{i}"
            )(x)
            x = ReLU(name=f"decoder_relu_{i}")(x)
            x = BatchNormalization(name=f"decoder_bn_{i}")(x)
        
        x = Conv2DTranspose(
            filters=1,
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name="decoder_conv_transpose_layer_final"
        )(x)
        
        decoder_output = Activation("sigmoid", name="sigmoid_layer")(x)
        
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

    def call(self, inputs):
        mu, log_variance, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed

    def train_step(self, data):
        x = data
        
        with tf.GradientTape() as tape:
            # Encode and decode
            mu, log_variance, z = self.encoder(x)
            reconstructed = self.decoder(z)
            
            # Calculate losses
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(x, reconstructed),
                    axis=[1, 2]
                )
            )
            
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + log_variance - tf.square(mu) - tf.exp(log_variance),
                    axis=1
                )
            )
            
            total_loss = self.reconstruction_loss_weight * reconstruction_loss + kl_loss

        # Calculate gradients and apply them
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def compile(self, learning_rate=0.0001):
        super(VAE, self).compile()
        self.optimizer = Adam(learning_rate=learning_rate)

    def save(self, save_folder="."):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            
        # Save parameters
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]
        with open(os.path.join(save_folder, "parameters.pkl"), "wb") as f:
            pickle.dump(parameters, f)
            
        # Save weights
        self.save_weights(os.path.join(save_folder, "weights.weights.h5"))

    def load_weights(self, weights_path):
        """Load weights for both encoder and decoder"""
        super(VAE, self).load_weights(weights_path)  #load_weights is a built-in function in keras

    def reconstruct(self, images):
        """Reconstruct images through the VAE"""
        mu, log_variance, latent_representations = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images, latent_representations

    @classmethod
    def load(cls, save_folder="."):
        """Load the VAE model from a folder"""
        # Load parameters
        with open(os.path.join(save_folder, "parameters.pkl"), "rb") as f:
            parameters = pickle.load(f)
            
        # Create a new VAE instance
        vae = cls(*parameters)
        
        # Build the model by running a forward pass
        dummy_input = np.zeros((1,) + parameters[0])  # parameters[0] is input_shape
        _ = vae(dummy_input)  # This builds the model
        
        # Load weights
        weights_path = os.path.join(save_folder, "weights.weights.h5")
        if os.path.exists(weights_path):
            vae.load_weights(weights_path)
        else:
            print(f"Warning: Weights file not found at {weights_path}")
        
        return vae

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()

if __name__ == "__main__":
    variationalautoencoder = VAE(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3 ,3),
        conv_strides=(1, 2, 2, 1),  #stride of 2 means that the output will be half the size of the input
                                    #stride of 1 means that the output will be the same size as the input
                                    #in other words, 2 means we are downsampling the data

        latent_space_dim=2    #we set this to 2 as we want to visualize the data in 2D
    )

    variationalautoencoder.summary()