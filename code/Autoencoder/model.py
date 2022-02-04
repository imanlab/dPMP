from tensorflow.keras.layers import Activation, Conv2DTranspose,Reshape, Flatten ,BatchNormalization, LeakyReLU, Dense, Input, Conv2D, MaxPooling2D, UpSampling2D
import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
def autoencoder(input_shape=(128, 128, 3)):
    '''
    AUTOENCODER MODEL DEFINITION
    '''
    '''
    ENCODER
    '''
    input_img = tf.keras.Input(shape=input_shape, name="img")

    x = Conv2D(32, (5, 5), strides =(2,2), padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(axis=-1)(x)

    x = Conv2D(64, (3, 3), strides =(2,2), padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(axis=-1)(x)

    x = Conv2D(128, (3, 3), strides =(2,2), padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(axis=-1)(x)

    volumeSize = K.int_shape(x)
    flatten = Flatten()(x)
    encoded = Dense(256,name = 'bottleneck')(flatten)
    encoded_representation = tf.keras.Model(input_img, encoded, name="encoder")
    '''
    DECODER
    '''
    x = Dense(np.prod(volumeSize[1:]))(encoded)
    x = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(x)

    x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(axis=-1)(x)

    x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(axis=-1)(x)

    x = Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(axis=-1)(x)

    x = Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(axis=-1)(x)

    x = Conv2DTranspose(3, 3, padding='same')(x)
    ae_decoder_output = Activation('sigmoid')(x)

    autoencoder = tf.keras.Model(input_img, ae_decoder_output,name="encoder_decoder")

    return encoded_representation, autoencoder

if __name__ =='__main__':
    encoded_representation,autoencoder = autoencoder()
    print(encoded_representation.summary())
    print(autoencoder.summary())
