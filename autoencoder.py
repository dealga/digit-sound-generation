from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, Reshape,Conv2DTranspose, Activation
#to get the shape of the data before flattening
#we want to move from flat data to a 3D arrays
from tensorflow.keras import backend as K 
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import os
import pickle

class Autoencoder:

    #this is a deep convolutional autoencoder
    def __init__(self, input_shape, conv_filters, conv_kernels, conv_strides, latent_space_dim):
        #latent space is the bottleneck of the autoencoder

        self.input_shape = input_shape #width, height, no. of channels
        self.conv_filters = conv_filters 
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim #bottleneck (if 2 then bottleneck will have 2 dimensions)

        self.encoder = None
        self.decoder = None
        self.model = None

        self._num_conv_layers = len(conv_filters)
        self.shape_before_bottleneck = None
        self._model_input = None


        self._build()


    def summary(self):
        self.encoder.summary()       # summary() is a built-in function in keras
        self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate=0.0001): #keras models require compiling before training
        optimizer = Adam(learning_rate=learning_rate)
        mse_loss = MeanSquaredError()
        self.model.compile(optimizer=optimizer, loss=mse_loss)  #self.model is the autoencoder
    
    def train(self, x_train, batch_size, num_epochs):
        self.model.fit(
            x_train, x_train, #the second x_train is the target i.e. we are trying to reconstruct the input
            batch_size=batch_size,                   
            epochs=num_epochs, 
            shuffle=True                                                 
            )     

    def save(self, save_folder = "."):
        self._create_folder_if_it_doesnt_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)   

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)  #load_weights is a built-in function in keras

    def reconstruct(self, images):
        latent_representations = self.encoder.predict(images)  #since encoder is a keras model, 
                                                               #we can use the in-built predict function
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images, latent_representations

       

    @classmethod #he load() method is used to create a new Autoencoder object based on some 
                #saved data (parameters and weights). It doesn't operate on an existing instance 
                # of Autoencoder. Instead, it creates a new instance by using saved information.
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, 'rb') as f:
            parameters = pickle.load(f)

        autoencoder = Autoencoder(*parameters) 
        weights_path = os.path.join(save_folder, "weights.weights.h5")
        autoencoder.load_weights(weights_path)

        return autoencoder 
    
            

    def _create_folder_if_it_doesnt_exist(self, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]

        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, 'wb') as f:
            pickle.dump(parameters, f)


    def _save_weights(self, save_folder):

        save_path = os.path.join(save_folder, "weights.weights.h5")
        self.model.save_weights(save_path)
        print(f"Saved weights to {save_path}")


    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_encoder(self):
        encoder_input = self._add_encoder_input() #add input layer to the encoder
        conv_layers = self._add_conv_layers(encoder_input) #add conv layers to the encoder
        bottleneck = self._add_bottleneck(conv_layers)  #bottleneck is the output of the encoder

        self._model_input = encoder_input #input to the model is the input to the encoder
                                          #this is used in the _build_autoencoder function
        self.encoder = Model(encoder_input, bottleneck, name="encoder")    #we are creating a keras model
                                                                           #and assigning it to the encoder
                                                                           #which was set as null in the 
                                                                           #constructor

    def _add_encoder_input(self):
        return Input(shape = self.input_shape, name="encoder_input")

    def _add_conv_layers(self, encoder_input): #this function creates all
                                              #convolutional blocks in the encoder
                                              #add conv_layer to each block
        #creates all convolutional blocks in encoder

        x = encoder_input

        #loop through all the conv layers
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x) #add conv layer to the graph of layers

        return x

    def _add_conv_layer(self, layer_index, x): #this function adds a convolutional block 
                                               #to a graph of layers
        # adds a convolutional block to a graph of layers, 
        # consisting of conv 2d + ReLU + batch normalization

        layer_number = layer_index + 1
        conv_layer = Conv2D(
            filters = self.conv_filters[layer_index],
            kernel_size = self.conv_kernels[layer_index],
            strides = self.conv_strides[layer_index],
            padding = "same",
            name = f"encoder_conv_layer_{layer_number}"
        )

        x = conv_layer(x)
        x = ReLU(name = f"encoder_relu_{layer_number}")(x)
        x = BatchNormalization(name = f"encoder_bn_{layer_number}")(x)

        return x

    def _add_bottleneck(self, x):

        #we store the shape of the data before flattening to use in the { DECODER }
        #this is done to mirror the encoder in the decoder

        self.shape_before_bottleneck = K.int_shape(x)[1:]   #[batch_size, 4, 4, 8] 
                                                            #we are interested in the 4, 4, 8
                                                        #which is the shape of the data before flattening



        
        #flatten data and add bottleneck(dense layer)
        x = Flatten(name="encoder_flatten")(x)
        x = Dense(self.latent_space_dim, name="encoder_output")(x) # here latent space dim is the number of
                                                                   # neurons in the dense layer
        return x

    def _build_decoder(self):
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

    def _add_decoder_input(self):
        return Input(shape=(self.latent_space_dim,), name="decoder_input")

    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self.shape_before_bottleneck) #this is the number of neurons in the dense layer
                                                         #suppose the shape is [4, 4, 8] 
                                                         # then the number of neurons is 4*4*8 = 128
    
        dense_layer = Dense(
            num_neurons, name="decoder_dense"
        )(decoder_input)

        return dense_layer

    def _add_reshape_layer(self, dense_layer):
        reshape_layer = Reshape(self.shape_before_bottleneck)(dense_layer)
        return reshape_layer



    def _add_conv_transpose_layers(self, x):
        #add conv transpose blocks
        #loop through all the conv layers in the encoder in reverse order
        for layer_index in reversed(range(1, self._num_conv_layers)):
            #we go in reverse order because we are going from the bottleneck to the output
            #we are mirroring the encoder in the decoder
            #we stop at 1 because we don't want to include the first layer
            #the first layer is the input layer

            x = self._add_conv_transpose_layer(layer_index, x)

        return x

    def _add_conv_transpose_layer(self, layer_index, x):
        layer_num = self._num_conv_layers - layer_index #_num_conv_layers is the total number of layers 
        conv_transpose_layer = Conv2DTranspose(
            filters = self.conv_filters[layer_index],
            kernel_size = self.conv_kernels[layer_index],
            strides = self.conv_strides[layer_index],
            padding = "same",
            name = f"decoder_conv_transpose_layer_{layer_num}" #layer_index is in reverse order 
                                                               #so we use layer_num
        )

        x = conv_transpose_layer(x)   #x is the input to the layer i.e. graph of layers

        x = ReLU(name = f"decoder_relu_{layer_num}")(x) 
        x = BatchNormalization(name = f"decoder_bn_{layer_num}")(x)

        return x
    
    def _add_decoder_output(self, x):
        #output layer
        conv_transpose_layer = Conv2DTranspose(
            filters = 1, #1 filter because we are reconstructing the original image
                         # input image is a grayscale image i.e. a SPECTOGRAM
            kernel_size = self.conv_kernels[0],
            strides = self.conv_strides[0],
            padding = "same",
            name = f"decoder_conv_transpose_layer_{self._num_conv_layers}" #self._num_conv_layers is the 
                                                             #total number of conv layers
        )

        x = conv_transpose_layer(x)
        output_layer = Activation("sigmoid", name="sigmoid")(x) 

        return output_layer



    def _build_autoencoder(self):
        #connect the encoder and decoder
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input)) #we feed model_input to the encoder

        self.model = Model(model_input, model_output, name="autoencoder") #model is the autoencoder

    


if __name__ == "__main__":
    autoencoder = Autoencoder(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3 ,3),
        conv_strides=(1, 2, 2, 1),  #stride of 2 means that the output will be half the size of the input
                                    #stride of 1 means that the output will be the same size as the input
                                    #in other words, 2 means we are downsampling the data

        latent_space_dim=2    #we set this to 2 as we want to visualize the data in 2D
    )

    autoencoder.summary()

