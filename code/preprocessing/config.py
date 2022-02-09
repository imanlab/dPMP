from datetime import datetime
import os,inspect

'''Experiment directory.'''
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
''' Code directory '''
CODE_DIR = ROOT_DIR +'/code/'
DATASET_DIR = ROOT_DIR +'/dataset/'
'''Directory of the collected dataset'''
COLLECTION_DIR = ROOT_DIR +'/rtp_multiple_views/'

''' Dataset directory '''
TRAJ_DIR = ROOT_DIR +'/dataset/trajectories/'
RGB_DIR = ROOT_DIR +'/dataset/rgb/'
RGB_DIR_T = ROOT_DIR +'/dataset/rgb_t/'
ADDITIONAL_DIR = ROOT_DIR +'/dataset/additional_images/'
ADDITIONAL_SEGMENTED = ROOT_DIR +'/dataset/additional_segmented/'
SEGMENTED = ROOT_DIR +'/dataset/segmented/'
DEPTH_DIR = ROOT_DIR +'/dataset/depth/'
ANNOTATION_PATH = ROOT_DIR +'/dataset/annotation/'
PLOT_DATA_PATH =ROOT_DIR +'/dataset/plot_distributions/'
BBOX_DIR = ROOT_DIR +'/dataset/bbox_annotations/'
RGB_ANNOTATED_DIR = ROOT_DIR +'/dataset/rgb_annotated/'
DETECTRON_DIR = CODE_DIR +'/Detectron/'
TASK_SPACE_ANNOTATION = ROOT_DIR +'/dataset/annotation_task/'
PLOT_TASK = ROOT_DIR +'/dataset/plot_task/'