import json
import os
from abc import ABC, abstractmethod
import numpy as np
from natsort import natsorted
from skimage.io import imread
from skimage.transform import resize
np.set_printoptions(threshold=np.inf)

def get_X_Y(filename):
    '''
    Gets the X,Y coordinate of the bbox
    '''
    X = []
    Y = []

    if filename.split(sep=".")[-1]== 'png':
        X.append(float(filename.split(sep='/')[-1].strip('.png').split(sep='_')[5]))
        Y.append(float(filename.split(sep='/')[-1].strip('.png').split(sep='_')[6]))
    Y = np.asarray(Y)/480
    X = np.asarray(X)/640

    return X,Y

def get_batch_X_Y(data_dir):
    '''
    Gets a list of X,Y coordinate of the bbox
    '''
    X = []
    Y = []
    for filename in natsorted(os.listdir(data_dir)):
        x,y = get_X_Y(filename)
        X.append(x)
        Y.append(y)
    X = [i for i in X if i.shape == (1,)]
    Y = [i for i in Y if i.shape == (1,)]
    Y = np.asarray(Y)
    X = np.asarray(X)
    return X,Y

def load_image(img_path, resize_shape=(128, 128, 3)):
    '''
    Loads the original RGB image
    '''
    img = imread(img_path, pilmode='RGB')
    if resize_shape:
        img = resize(img, resize_shape)
    return img.astype('float32')

def load_images(ids, img_path, resize_shape=(128, 128, 3)):
    '''
    Loads a set of RGB images (the ones listed in images ids)
    '''
    return np.array([load_image(os.path.join(img_path, id + ".png"), resize_shape) for id in ids])

def load_encoded_image(img_path, encoder, resize_shape=(128, 128, 3)):
    '''
    Loads an encoded image
    '''

    latent_mean,latent_var,latent_sample = encoder(np.expand_dims(load_image(img_path, resize_shape=resize_shape), axis=0))
    return np.squeeze(latent_mean)

def load_encoded_images(ids, img_dir, encoder, resize_shape=(128, 128, 3)):
    '''
    Loads a set of encoded images (the ones listed in images ids)
    '''
    return np.array([load_encoded_image(os.path.join(img_dir, id + ".png"), encoder, resize_shape) for id in ids])


def get_id(filename):
    '''
    Gets  the id of a .json file
    '''
    if filename.split(sep=".")[-1]== 'png':
     return filename.split(sep='/')[-1].strip('.png').split(sep='_')[0]

def get_ids(data_dir):
    '''
    Gets a list of ids of the .json files
    '''
    out=np.array(natsorted([get_id(filename) for filename in os.listdir(data_dir)]))
    out = [i for i in out if i!= None]
    out=np.asarray(out)
    return out

def get_images_id(filename):
    '''
    Gets the single id of an image.
    '''
    if filename.split(sep=".")[-1]== 'png':
     return filename.split(sep='/')[-1].strip('.png')

def get_images_ids(data_dir):
    '''
    Gets the list of ids of images in a directory
    '''
    out=np.array(natsorted([get_images_id(filename) for filename in os.listdir(data_dir)]))
    out = [i for i in out if i!= None]
    out=np.asarray(out)
    return out

def load_weights_mean(json_path, mean_key="mean_weights"):
    '''
    Loads the mean weights saved in a .json files as a dictionary with key mean_key
    '''
    with open(json_path, 'r') as fp:
        annotation = json.load(fp)
        return np.asarray(annotation[mean_key]).round(16).astype('float64')

def load_batch_weights_mean(dataset_path, ids):
    '''
    loads a batch of mean of weights
    '''
    return np.asarray([load_weights_mean(os.path.join(dataset_path, id + ".json")) for id in ids], dtype=object)   #(506, 56, 1)

def load_weights_L(json_path, L_key="L"):
    '''
    Loads the non zero values of the Cholesky decomposition of the covariance matrix saved in a .json files as a dictionary with key L.
    '''
    with open(json_path, 'r') as fp:
        annotation = json.load(fp)
        return np.asarray(annotation[L_key]).round(16).astype('float64')

def load_batch_weights_L(dataset_path, ids):
    '''
    Loads a batch of non zero values of the Cholesky decomposition of the covariance matrix
    '''
    return np.asarray([load_weights_L(os.path.join(dataset_path, id + ".json")) for id in ids], dtype=object)         #(506, 1596)



def get_number_id(filename):
    '''
    Gets the sample number
    '''

    if filename.split(sep=".")[-1]== 'png':
     return filename.split(sep='/')[-1].strip('.png').split(sep='_')[3]

def get_number_ids(data_dir):
    '''
    Gets a list of sample numbers
    '''
    out=np.array([get_number_id(filename) for filename in natsorted(os.listdir(data_dir))])
    out = [i for i in out if i != None]
    out = np.asarray(out)
    return out
def get_berry_id(filename):
    '''
    Gets the sample number
    '''

    if filename.split(sep=".")[-1]== 'png':
     return filename.split(sep='/')[-1].strip('.png').split(sep='_')[4]

def get_berry_ids(data_dir):
    '''
    Gets a list of sample numbers
    '''
    out=np.array([get_berry_id(filename) for filename in natsorted(os.listdir(data_dir))])
    out = [i for i in out if i != None]
    out = np.asarray(out)
    return out
def get_config_id(filename):
    '''
    Gets the sample number
    '''

    if filename.split(sep=".")[-1]== 'png':
     return filename.split(sep='/')[-1].strip('.png').split(sep='_')[1].strip('conf')

def get_config_ids(data_dir):
    '''
    Gets a list of sample numbers
    '''
    out = np.array([get_config_id(filename) for filename in natsorted(os.listdir(data_dir))])
    out = [i for i in out if i != None]
    out = np.asarray(out)
    return out

class Dataset(ABC):
    """
    The base dataset only loads the IDs of the files and the mean and L of the ProMPs weights
    """
    def __init__(self, dataset_dir,rgb_dir):
        self.samples = {}
        self.dataset_dir = dataset_dir
        self.rgb_dir = rgb_dir
        # Load all IDs, images IDs,mean of weights and fake L elements
        self.samples["id"] = get_ids(self.rgb_dir)                                                   #  ['0' '0' '0' ... '24' '24' '24']
        self.samples["images_id"] = get_images_ids(self.rgb_dir)                                     #  ['0_conf1_1_000_berry1_Xcenter_Ycenter' '0_conf1_1_001_berry1_Xcenter_Ycenter' '0_conf1_1_002_berry1_Xcenter_Ycenter' ... '24_conf5_t_007_berry5_Xcenter_Ycenter' '24_conf5_t_008_berry5_Xcenter_Ycenter' '24_conf5_t_009_berry5_Xcenter_Ycenter']
        self.samples["number_id"] = get_number_ids(self.rgb_dir)                                     #  ['000' '001' '002' ... '007' '008' '009']
        self.samples["mean_weights"] = load_batch_weights_mean(self.dataset_dir, self.samples["id"])
        self.samples["L"] = load_batch_weights_L(self.dataset_dir, self.samples["id"])
        self.samples["config"] = get_config_ids(self.rgb_dir)
        self.samples["berries_id"] = get_berry_ids(self.rgb_dir)
        self.samples["X"],self.samples["Y"] = get_batch_X_Y(self.rgb_dir)


    @abstractmethod
    def _split_train_test_val(self):
        """Return the list of indexes corresponding to training, validation and testing."""
        raise NotImplementedError()

    @abstractmethod
    def prepare_data(self):
        """Prepare the data from self.samples to self.data in the format (X_train, y_train), (X_val, y_val), (X_test, y_test)."""
        raise NotImplementedError()

class DatasetOrdinary(Dataset, ABC):

    def __init__(self, rgb_dir,dataset_dir, **kwargs):
        super().__init__(rgb_dir=rgb_dir,dataset_dir=dataset_dir, **kwargs)

    def _split_train_test_val(self):
        idx_all = np.arange(0, self.samples["id"].shape[0])
        idx_train = np.array([], dtype=np.int64)
        idx_test = np.array([], dtype=np.int64)
        idx_val = np.array([], dtype=np.int64)

        for index in idx_all:
            if self.samples["config"][index] == '5' and self.samples["berries_id"][index] == 'berry3':
                idx_test = np.append(idx_test,index)
                idx_val = np.append(idx_val,index)
            else:
                idx_train = np.append(idx_train,index)
        return idx_train, idx_val, idx_test



class DatasetRGB(Dataset, ABC):
    """
    RGB datasets load the encoded images in the samples.
    """
    def __init__(self, dataset_dir, rgb_dir, encoder,**kwargs):
        self.rgb_dir = rgb_dir
        self.encoder = encoder

        super().__init__(dataset_dir = dataset_dir,rgb_dir = rgb_dir, **kwargs)
        self.samples["img_enc"] = load_encoded_images(self.samples["images_id"], self.rgb_dir,self.encoder)



