import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Concatenate
import tensorflow_probability as tfp
tfd = tfp.distributions
tf.keras.backend.set_floatx('float64')

class deep_ProMPs_2dmodel_RGBD(tf.keras.Model):

    def __init__(self):
        super(deep_ProMPs_2dmodel_RGBD, self).__init__()

        # Define layers.
        self.x1 = layers.Dense(16)
        self.x2 = layers.LeakyReLU(0.2)
        self.x3 = layers.Dense(8)
        self.x4 = layers.LeakyReLU(0.2)

        # Mean prediction.
        self.xa = tf.keras.layers.Dense(units = 128, activation="tanh", kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0))
        self.xb = tf.keras.layers.Dense(units = 64, activation="tanh", kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0))
        self.xc = tf.keras.layers.Dense(units = 32, activation="tanh", kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0))
        self.xd = tf.keras.layers.Dense(units = 16, activation="tanh", kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0))
        self.mean_weights = tf.keras.layers.Dense(units = 8, activation="linear")

        #Fake L prediciton
        self.xe = tf.keras.layers.Dense(units = 128, activation="tanh", kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0))
        self.xf = tf.keras.layers.Dense(units = 64, activation="tanh", kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0))
        self.fakeL = tf.keras.layers.Dense(units = 36, activation="linear")

        self.concatenate = Concatenate(axis=1)

    def call(self, inputs):

        x = self.x1(inputs)
        x = self.x2(x)
        x = self.x3(x)
        x = self.x4(x)
        # 1 branch - Mean prediction
        x1 = self.xa(x)
        x1 = self.xb(x1)
        x1 = self.xc(x1)
        x1 = self.xd(x1)
        mean_weights = self.mean_weights(x1)    # (8)
        # 2 branch - Covariance prediction
        x2 = self.xe(x)
        x2 = self.xf(x2)
        fakeL = self.fakeL(x2)                  # (36)

        return self.concatenate([mean_weights, fakeL])

    def summary(self):
        encoded = layers.Input(shape=(256,))
        model = tf.keras.Model(inputs=encoded, outputs=self.call(encoded))

        return model.summary()

if __name__ == "__main__":
    model = deep_ProMPs_2dmodel_RGBD()
    model.summary()