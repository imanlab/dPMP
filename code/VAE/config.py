import time, os,inspect
#MAIN ROOT
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
dataset_dir =os.path.dirname(os.path.dirname(current_dir)) +"/dataset/"
print(current_dir)
#LOSS PLOT DIRECTORY
PLOT_PATH =  current_dir + "/LOSS_VAE"
#IMG DIRECTORY
IMG_DIR = dataset_dir + "rgb_tot_white/"
#MODEL SAVE DIRECTORY
ENCODED_PATH_ENCODER = current_dir + "/MODEL_VAE"
ENCODED_PATH_DECODER = current_dir + "/DECODER"
#BOTTLENECK IMAGES DIRECTORY
BOTTLENECK_PATH = current_dir + "/BOTTLENECK IMAGES VAE"
#RECONTRUCTED IMAGES DIRECTORY
IMAGES = current_dir + "/RECONSTRUCTED IMAGES VAE"
