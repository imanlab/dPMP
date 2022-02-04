from tensorflow.keras.layers import Activation,Concatenate,Dense, Conv2DTranspose,LeakyReLU,BatchNormalization,Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import backend as K

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def autoencoder(input_shape=(128, 128, 3),encoded_shape=256):
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

    encoded_mean = Dense(encoded_shape, name="bottleneck_mean")(flatten)
    encoded_var = Dense(encoded_shape, name="bottleneck_var")(flatten)

    z = Sampling()([encoded_mean, encoded_var])
    encoder = tf.keras.Model(inputs=input_img, outputs=[encoded_mean, encoded_var,z], name="encoder")

    '''
    DECODER
    '''
    encoded_img = tf.keras.Input(shape=(256,), name="img")
    x = Dense(np.prod(volumeSize[1:]))(encoded_img)
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

    decoder = tf.keras.Model(inputs=encoded_img, outputs=ae_decoder_output, name="decoder")
    return encoder, decoder

class CVAE(keras.Model):
    def __init__(self,input_shape,encoded_shape, **kwargs):
        super(CVAE, self).__init__(**kwargs)
        self.encoder,self.decoder = autoencoder(input_shape=input_shape,encoded_shape=encoded_shape)
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        self.total_loss_tracker_val = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker_val = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker_val = keras.metrics.Mean(name="kl_loss")

    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            img = data
            z_mean, z_log_var,z = self.encoder(img)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(img, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    @tf.function
    def test_step(self, data_val):
        img_val = data_val
        z_mean_val, z_log_var_val, z_val = self.encoder(img_val)
        reconstruction = self.decoder(z_val)
        reconstruction_loss_val = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(img_val, reconstruction), axis=(1, 2)
            )
        )
        kl_loss_val = -0.5 * (1 + z_log_var_val - tf.square(z_mean_val) - tf.exp(z_log_var_val))
        kl_loss_val = tf.reduce_mean(tf.reduce_sum(kl_loss_val, axis=1))
        total_loss_val = reconstruction_loss_val + kl_loss_val

        self.total_loss_tracker_val.update_state(total_loss_val)
        self.reconstruction_loss_tracker_val.update_state(reconstruction_loss_val)
        self.kl_loss_tracker_val.update_state(kl_loss_val)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {
            "loss": self.total_loss_tracker_val.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker_val.result(),
            "kl_loss": self.kl_loss_tracker_val.result(),
        }

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    def __call__(self,data, **kwargs):
        img = data
        z_mean, z_log_var, z = self.encoder(img)
        reconstruction = self.decoder(z)
        return reconstruction

if __name__ == "__main__":
    cvae = CVAE(input_shape=(128, 128, 3),encoded_shape=256)
    cvae.decoder.summary()
    cvae.encoder.summary()
