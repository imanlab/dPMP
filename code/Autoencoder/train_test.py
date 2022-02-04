from tensorflow import keras
import time, sys
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
import matplotlib.pyplot as plt
from datagenerator import *
from datagenerator import load_images
import imageio
from config import PLOT_PATH, IMG_DIR ,ENCODED_PATH, IMAGES
from pathlib import Path
from model import autoencoder


def train(img_dir,loss_path,model_path,batch_size = 32):
    '''
    Function that does the training od the autoencoder
    :param img_dir: path of the dataset of images
    :param loss_path: path to save the loss function graph
    :param model_path: path to save the model
    :param batch_size: batch size
    '''
    print("Loading data...")
    dataset = DatasetRGB(rgb_dir=img_dir)
    dataset.prepare_data()
    print("Done!")
    (X_train), (X_val), (_) = dataset.data
    RGB_train = X_train["RGB"]
    RGB_val = X_val["RGB"]
    print('RGB train :   ', np.shape(RGB_train))
    print('RGB val :   ', np.shape(RGB_val))

    history = autoencoder.fit(RGB_train,RGB_train,epochs=1500,batch_size=batch_size,validation_data =(RGB_val,RGB_val),shuffle=True)
    #SAVE THE HISTORY PLOT
    plt.plot(history.history['loss'], 'b')
    plt.title('cnnautoencoder_mse_{}'.format(autoencoder.optimizer._name))
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.legend(['mse'], loc='upper right')
    plt.savefig(loss_path + '/cnnautoencoder_mse_{}.png'.format(time.time()))

    #SAVE THE MODEL
    autoencoder.save(model_path)

def test(encoded_path,img_dir,reconstructed_images_path):
    '''
    Funciton that tests the autoencoder
    :param encoded_path: path of the autoecoder model
    :param img_dir_test: path of the images to be tested
    :param reconstructed_images_path: path to save the reconstructed images+ the input images
    '''
    print("Loading data...")
    dataset = DatasetRGB(rgb_dir = img_dir)
    dataset.prepare_data()
    print("Done!")

    (_), (_), (X_test) = dataset.data
    RGB_test = X_test["RGB"]

    autoencoder = tf.keras.models.load_model(encoded_path)

    decoded_imgs = autoencoder.predict(RGB_test)  # reconstruction
    encoder = Model(autoencoder.input, autoencoder.get_layer('bottleneck').output)
    latent_dim = encoder.predict(RGB_test)

    for x, y in enumerate(decoded_imgs):
        orig = RGB_test[x]
        img = np.hstack((orig, y))
        imageio.imwrite(reconstructed_images_path+ "/img_{}.jpg".format(x+1), img)

if __name__ == "__main__":

    '''
    TRAIN THE AUTOENCODER
    '''
    # CREATE DIRECTORIES
    Path(PLOT_PATH).mkdir(exist_ok=True, parents=True)
    Path(ENCODED_PATH).mkdir(exist_ok=True, parents=True)
    Path(IMAGES).mkdir(exist_ok=True, parents=True)
    # MODEL DEFINITION
    encoded_representation,autoencoder = autoencoder(input_shape=(128, 128, 3))
    encoded_representation.summary()
    autoencoder.summary()
    autoencoder.compile(optimizer= keras.optimizers.Adam(), loss='mse')
    # TRAINING
    train(batch_size=32, img_dir=IMG_DIR, loss_path=PLOT_PATH, model_path=ENCODED_PATH)


    '''
    TEST THE AUTOENCODER
    '''
    test(encoded_path=ENCODED_PATH,img_dir=IMG_DIR,reconstructed_images_path=IMAGES)