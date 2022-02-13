from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from datagenerator import *
import imageio
from config import PLOT_PATH, IMG_DIR ,ENCODED_PATH_ENCODER,ENCODED_PATH_DECODER,  IMAGES
from pathlib import Path
from model import CVAE,autoencoder
from datagenerator import DatasetRGB
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

def train(img_dir,loss_path,model_path_encoder,model_path_decoder,batch_size = 32):
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
   
    cvae.encoder.save(model_path_encoder)
    cvae.decoder.save(model_path_decoder)


def test(model_path_encoder,model_path_decoder,img_dir,reconstructed_images_path):
    '''
    Funciton that tests the autoencoder
    :param encoded_path: path of the autoecoder model
    :param img_dir_test: path of the images to be tested
    :param bottleneck_path: path to save the bottleneck images
    :param reconstructed_images_path: path to save the reconstructed images+ the input images
    '''
    encoder = tf.saved_model.load(model_path_encoder)
    decoder = tf.saved_model.load(model_path_decoder)
    print("Loading data...")
    dataset = DatasetRGB(rgb_dir = img_dir)
    dataset.prepare_data()
    print("Done!")

    (_), (_), (X_test) = dataset.data
    RGB_test = X_test["RGB"]

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
    Path(ENCODED_PATH_ENCODER).mkdir(exist_ok=True, parents=True)
    Path(ENCODED_PATH_DECODER).mkdir(exist_ok=True, parents=True)
    Path(IMAGES).mkdir(exist_ok=True, parents=True)
    # TRAINING
    train(batch_size=32, img_dir=IMG_DIR, loss_path=PLOT_PATH, model_path_encoder=ENCODED_PATH_ENCODER, model_path_decoder=ENCODED_PATH_DECODER)
    '''
    TEST THE AUTOENCODER
    '''
    test(model_path_encoder=ENCODED_PATH_ENCODER, model_path_decoder=ENCODED_PATH_DECODER,img_dir=IMG_DIR,reconstructed_images_path=IMAGES)
