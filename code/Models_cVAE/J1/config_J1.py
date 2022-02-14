from datetime import datetime
import os,inspect

''' FOLDERS DEFINITNION '''
''' Define date. '''
now = datetime.now().strftime("%d_%m__%H_%M")
''' Experiment directory. '''
J_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(J_DIR)))

'''Data.'''

'''RGB-D images folder. '''
IMAGE_PATH = os.path.join(ROOT_DIR, "dataset/rgb_segmented_white/")


''' Annotations folder. '''
ANNOTATION_PATH = os.path.join(ROOT_DIR,"dataset/annotation/J1")

''' Save the model '''
MODEL_FOLDER= os.path.join(J_DIR, "MODELS")
MODEL_NAME = f"model_{now}"
MODEL_PATH = os.path.join(MODEL_FOLDER, MODEL_NAME)

''' ProMPs deep model outputs '''
OUTPUT_PATH= os.path.join(J_DIR, "PLOTS")
LOSS_PATH = os.path.join(J_DIR, "LOSS", MODEL_NAME)
METRIC_PATH = os.path.join(J_DIR, "METRIC")
LOG_DIR = os.path.join(J_DIR, "LOGS", MODEL_NAME)

''' Encoder path '''
ENCODER_MODEL_PATH = os.path.join(ROOT_DIR,"code/VAE/MODEL_VAE")

''' HYPERPARAMETERS. '''
# Batch sze.
batch = 5
# Number of epochs
epochs = 2000
# Learning rate
lr = 0.0001
# Seed of the random state.
random_state = 42
# Kernel regularization.
l1_reg = 0
l2_reg = 0
# Validation set percentage
val_frac = 0.2
# Wether to use validation samples during training
use_val_in_train = False
# Number of test samples.
N_test = 7

''' CALLBACKS '''
# Early stopping
es = {"enable":False,
      "delta": 0,
      "patience": 40,
      "restore_best_weight":True}
# Reduce learning rate
rl = {"enable":True,
      "factor": 0.99,
      "min_lr": 0.000000001,
      "patience": 5}
