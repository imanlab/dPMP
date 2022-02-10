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

        # Define layers x,y

        self.X1 = layers.Dense(64)
        self.X2 = layers.LeakyReLU(0.2)
        self.Y1 = layers.Dense(64)
        self.Y2 = layers.LeakyReLU(0.2)


        # Define layers img
        self.x1 = layers.Dense(128)
        self.x2 = layers.LeakyReLU(0.2)
        # self.x3 = layers.Dense(32)
        # self.x4 = layers.LeakyReLU(0.2)

        # Mean prediction.
        self.xa = tf.keras.layers.Dense(units = 128, activation="tanh", kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0))
        self.xb = tf.keras.layers.Dense(units = 64, activation="tanh", kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0))
        self.xc = tf.keras.layers.Dense(units = 32, activation="tanh", kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0))
        self.xd = tf.keras.layers.Dense(units = 16, activation="tanh", kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0))
        self.mean_weights = tf.keras.layers.Dense(units = 8, activation="linear")

        # Fake L prediciton
        self.xe = tf.keras.layers.Dense(units = 128, activation="tanh", kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0))
        self.xf = tf.keras.layers.Dense(units = 64, activation="tanh", kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.0, l2=0.0))
        self.fakeL = tf.keras.layers.Dense(units = 36, activation="linear")

        self.concatenate = Concatenate(axis=1)
        self.concatenate_xy_img = Concatenate(axis=1)

    def call(self, inputs):
        img = inputs[0]
        x = inputs[1]
        y = inputs[2]

        img = self.x1(img)
        img = self.x2(img)
        # img = self.x3(img)
        # img = self.x4(img)

        x = self.X1(x)
        x = self.X2(x)

        y = self.Y1(y)
        y = self.Y2(y)

        xyimg = self.concatenate_xy_img([x,y,img])
        # 1 branch - Mean prediction
        x1 = self.xa(xyimg)
        x1 = self.xb(x1)
        x1 = self.xc(x1)
        x1 = self.xd(x1)
        mean_weights = self.mean_weights(x1)    # (8)
        # 2 branch - Covariance prediction
        x2 = self.xe(xyimg)
        x2 = self.xf(x2)
        fakeL = self.fakeL(x2)                  # (36)

        return self.concatenate([mean_weights, fakeL])

    def summary(self):
        encoded = layers.Input(shape=(256,))
        x = layers.Input(shape=(1,))
        y = layers.Input(shape=(1,))
        model = tf.keras.Model(inputs=[encoded,x,y], outputs=self.call([encoded,x,y]))

        return model.summary()

if __name__ == "__main__":
    model = deep_ProMPs_2dmodel_RGBD()
    model.summary()