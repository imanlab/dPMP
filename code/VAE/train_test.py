from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from datagenerator import *
import imageio
from config import PLOT_PATH, IMG_DIR ,ENCODED_PATH,  IMAGES
from pathlib import Path
from model import CVAE,autoencoder
from datagenerator import DatasetRGB
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def train(img_dir,loss_path,model_path,batch_size = 32):
    '''
    Function that does the training od the autoencoder
    :param img_dir: path of the dataset of images
    :param loss_path: path to save the loss function graph
    :param model_path: path to save the model
    :param batch_size: batch size
    '''
    print("Loading data...")
    dataset = DatasetRGB( rgb_dir=img_dir)
    dataset.prepare_data()
    print("Done!")
    (X_train), (X_val), (_) = dataset.data

    RGB_train = X_train["RGB"]
    RGB_val = X_val["RGB"]


    print('RGB train:   ', np.shape(RGB_train))
    print('RGB val:   ', np.shape(RGB_val))

    cvae = CVAE(input_shape=(128, 128, 3),encoded_shape=256)
    cvae.compile(optimizer = keras.optimizers.Adam())
    history = cvae.fit(RGB_train, epochs=1500, batch_size=batch_size,validation_data =(RGB_val,None),shuffle=True)

    #SAVE THE HISTORY PLOT
    plt.figure()
    plt.plot(history.history['loss'], 'b')
    plt.plot(history.history['val_loss'], 'r')
    plt.xlabel('epoch')
    plt.savefig(loss_path + '/VAE_loss_.png')

    #SAVE THE HISTORY PLOT
    plt.figure()
    plt.plot(history.history["kl_loss"], 'b')
    plt.plot(history.history['val_kl_loss'], 'r')
    plt.xlabel('epoch')
    plt.savefig(loss_path + '/VAE_kl_loss_.png')

    #SAVE THE HISTORY PLOT
    plt.figure()
    plt.plot(history.history['reconstruction_loss'], 'b')
    plt.plot(history.history['val_reconstruction_loss'], 'r')
    plt.xlabel('epoch')
    plt.savefig(loss_path + '/VAE_reconstruction_loss.png')

    #SAVE THE MODEL
    tf.saved_model.save(cvae, model_path)


def test(encoded_path,img_dir,reconstructed_images_path):
    '''
    Funciton that tests the autoencoder
    :param encoded_path: path of the autoecoder model
    :param img_dir_test: path of the images to be tested
    :param bottleneck_path: path to save the bottleneck images
    :param reconstructed_images_path: path to save the reconstructed images+ the input images
    '''
    cvae = tf.saved_model.load(encoded_path)
    print("Loading data...")
    dataset = DatasetRGB(rgb_dir = img_dir)
    dataset.prepare_data()
    print("Done!")

    (_), (_), (X_test) = dataset.data
    RGB_test = X_test["RGB"]


    encoder = cvae.encoder
    decoder = cvae.decoder


    latent_mean,latent_var,latent_sample = encoder(RGB_test)
    decoded_imgs = decoder(latent_mean)

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
    # TRAINING
    train(batch_size=32, img_dir=IMG_DIR, loss_path=PLOT_PATH, model_path=ENCODED_PATH)
    '''
    TEST THE AUTOENCODER
    '''
    test(encoded_path=ENCODED_PATH,img_dir=IMG_DIR,reconstructed_images_path=IMAGES)
