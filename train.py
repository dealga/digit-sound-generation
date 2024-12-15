from tensorflow.keras.datasets import mnist
from autoencoder import Autoencoder


LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHS = 20

def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()#60,000 training images and 10,000 test images

    
    x_train = x_train.astype('float32') / 255 #we normalize the pixel values to be between 0 and 1
    x_train = x_train.reshape(x_train.shape + (1,)) #we add a channel dimension to the data
                                                    #to make it compatible with the convolutional layers
                                                    #i.e. default shape is (60000, 28, 28) and 
                                                    #we change it to (60000, 28, 28, 1)
                                                    #which indicates that it is greyscale
    x_test = x_test.astype('float32') / 255 
    x_test = x_test.reshape(x_test.shape + (1,))

    return x_train, y_train, x_test, y_test


def train(x_train, learning_rate, batch_size, epochs):
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

    autoencoder.compile(learning_rate)
    autoencoder.train(x_train, batch_size, epochs)

if __name__ == "__main__":

    x_train, _, _, _ = load_mnist()  #we put _ for the labels because we are not using them
    #x_train, y_train, x_test, y_test = load_mnist()

    autoencoder = train(x_train[:500],LEARNING_RATE, BATCH_SIZE, EPOCHS)