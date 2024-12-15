from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, Reshape,Conv2DTranspose, Activation
#to get the shape of the data before flattening
#we want to move from flat data to a 3D arrays
from tensorflow.keras import backend as K 
import numpy as np

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
        self._build()


    def summary(self):
        self.encoder.summary()       # summary() is a built-in function in keras
        #self.decoder.summary()

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        #self._build_autoencoder()

    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self.encoder = Model(encoder_input, bottleneck, name="encoder")

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


if __name__ == "__main__":
    autoencoder = Autoencoder(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3 ,3),
        conv_strides=(1, 2, 2, 1),  #stride of 2 means that the output will be half the size of the input
                                    #stride of 1 means that the output will be the same size as the input
                                    #in other words, 2 means we are downsampling the data
        latent_space_dim=2
    )

    autoencoder.summary()

