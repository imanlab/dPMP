import time, os,inspect
#MAIN ROOT
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
dataset_dir =os.path.dirname(os.path.dirname(current_dir)) +"/dataset/"
print(current_dir)
#LOSS PLOT DIRECTORY
PLOT_PATH =  current_dir + "/LOSS"
#IMG DIRECTORY
IMG_DIR = dataset_dir + "rgb_tot_white/"
#MODEL SAVE DIRECTORY
ENCODED_PATH = current_dir + "/MODEL"
#RECONTRUCTED IMAGES DIRECTORY
IMAGES = current_dir + "/RECONSTRUCTED IMAGES"
