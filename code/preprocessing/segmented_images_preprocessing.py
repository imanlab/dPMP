import pathlib
import re
from tqdm import tqdm
from pathlib import Path
import config as cfg
import os, json, cv2, random,sys
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
import numpy as np
import config as config
import re
numbers = re.compile(r'(\d+)')
from skimage.transform import resize


for d in ["train"]:
    MetadataCatalog.get("strawberry_" + d).set(thing_classes=["pluckable", "unpluckable"])
strawberry_metadata = MetadataCatalog.get("strawberry_train")
cfg = get_cfg()
cfg.MODEL.DEVICE='cpu'

cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 1100
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (strawberry).
cfg.MODEL.WEIGHTS = os.path.join(config.DETECTRON_DIR, "model_final.pth")  # path to the trained model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
predictor = DefaultPredictor(cfg)



def segmented_rgb(img_dir_ ,save_path):
        for idx, img_dir in enumerate(pathlib.Path(img_dir_).rglob('*.png')):
            print(idx, '   :   ', img_dir.name)
            im = cv2.imread(img_dir.as_posix())
            outputs = predictor(im)
            outputs = outputs['instances'].to('cpu')
            ripeness = []
            for j in range(np.size(outputs.pred_classes.numpy())):
                if outputs.pred_classes.numpy()[j] == 1:
                    ripeness.append('Unripe')
                else:
                    ripeness.append('Ripe')
            print('Ripeness:   ', ripeness)
            ''' Bounding box + segmentation mask extraction '''
            pred_box, pred_mask = outputs.pred_boxes.tensor.numpy(), outputs.pred_masks.numpy()
            ''' Ordering '''
            X_tot_center = []
            Y_tot_center = []

            for n_strawberry in range(pred_mask.shape[0]):
                if ripeness[n_strawberry] == 'Ripe':
                    x0 = pred_box[n_strawberry, 0]
                    x1 = pred_box[n_strawberry, 2]
                    y0 = pred_box[n_strawberry, 1]
                    y1 = pred_box[n_strawberry, 3]
                    X_center = x0 + (x1 - x0) / 2
                    Y_center = y1 + (y0 - y1) / 2

                    X_tot_center.append([n_strawberry, X_center])
                    Y_tot_center.append([n_strawberry, Y_center])

            X_tot_center = np.asarray(X_tot_center)
            ind = np.argsort(X_tot_center[:, 1])
            ordered_ripe_ind = []
            for k in range(1, ind.shape[0] + 1):
                ordered_ripe_ind.append(int(X_tot_center[ind[-k]][0]))
            ordered_ripe_ind = np.asarray(ordered_ripe_ind)
            print('Detected berries: ', X_tot_center.shape[0])
            assert X_tot_center.shape == (5,2)
            #Path(img_dir).rename(config.DATASET_DIR+'rgb_t_moved/'+img_dir.name)

            n_strawberry = int(img_dir.name.split(sep='_')[4].strip('berry').strip('.png'))

            n_berry = ordered_ripe_ind[n_strawberry - 1]
            ''' Bounding box'''
            x0 = pred_box[n_berry, 0]
            y0 = pred_box[n_berry, 1]
            x1 = pred_box[n_berry, 2]
            y1 = pred_box[n_berry, 3]
            X = x0 + (x1 - x0) / 2
            Y = y1 + (y0 - y1) / 2

            pred_mask_single = np.squeeze(pred_mask[n_berry, :, :])
            im_seg = im * np.dstack((pred_mask_single, pred_mask_single, pred_mask_single))

            for x in range(480):
                for y in range(640):
                    if im_seg[x,y].all() == 0:
                       im_seg[x,y][:] = 255

            cv2.circle(im_seg, (int(X),int(Y)), 2, (0, 0, 255), 10)
            cv2.rectangle(im_seg, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 255),2)
            img = resize(im, (256,256,3))
            cv2.imshow('Segmentation', img)
            key = cv2.waitKey()

            id = img_dir.name.split(sep='_')[0]+'_'+img_dir.name.split(sep='_')[1]+'_'+img_dir.name.split(sep='_')[2]+'_'+img_dir.name.split(sep='_')[3]
            save_name = id + '_berry'+str(int(n_strawberry))+'_'+str(round(X,3))+'_'+str(round(Y,3))+'.png'

            #cv2.imwrite(save_path + save_name, im_seg)
            print('Save name   :   ', save_name)

def segmented_rgb_detect(img_dir_ ,save_path):
        for idx, img_dir in enumerate(sorted(pathlib.Path(img_dir_).rglob('*.png'))):
            print(idx, '   :   ', img_dir.name)
            im = cv2.imread(img_dir.as_posix())
            outputs = predictor(im)
            outputs = outputs['instances'].to('cpu')
            ripeness = []
            for j in range(np.size(outputs.pred_classes.numpy())):
                if outputs.pred_classes.numpy()[j] == 1:
                    ripeness.append('Unripe')
                else:
                    ripeness.append('Ripe')
            print('Ripeness:   ', ripeness)
            ''' Bounding box + segmentation mask extraction '''
            pred_box, pred_mask = outputs.pred_boxes.tensor.numpy(), outputs.pred_masks.numpy()
            ''' Ordering '''
            X_tot_center = []
            Y_tot_center = []

            for n_strawberry in range(pred_mask.shape[0]):
                if ripeness[n_strawberry] == 'Ripe':
                    x0 = pred_box[n_strawberry, 0]
                    x1 = pred_box[n_strawberry, 2]
                    y0 = pred_box[n_strawberry, 1]
                    y1 = pred_box[n_strawberry, 3]
                    X_center = x0 + (x1 - x0) / 2
                    Y_center = y1 + (y0 - y1) / 2

                    X_tot_center.append([n_strawberry, X_center])
                    Y_tot_center.append([n_strawberry, Y_center])

            X_tot_center = np.asarray(X_tot_center)
            ind = np.argsort(X_tot_center[:, 1])
            ordered_ripe_ind = []
            for k in range(1, ind.shape[0] + 1):
                ordered_ripe_ind.append(int(X_tot_center[ind[-k]][0]))
            ordered_ripe_ind = np.asarray(ordered_ripe_ind)
            print('Detected berries: ', X_tot_center.shape[0])
            assert X_tot_center.shape == (5,2)
            n_strawberry =  img_dir.name.split(sep='_')[4].strip('berry').strip('.png')

            n_berry = ordered_ripe_ind[int(n_strawberry) - 1]
            ''' Bounding box'''
            x0 = pred_box[n_berry, 0]
            y0 = pred_box[n_berry, 1]
            x1 = pred_box[n_berry, 2]
            y1 = pred_box[n_berry, 3]
            X = x0 + (x1 - x0) / 2
            Y = y1 + (y0 - y1) / 2
            im = cv2.imread(img_dir.as_posix())
            cv2.rectangle(im, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 255),-1)
            # cv2.circle(im, (int(X),int(Y)), 2, (0, 0, 255), 10)
            # cv2.imshow('Segmentation', im)
            # key = cv2.waitKey()


            #id = img_dir.name.split(sep='_')[0]+'_'+img_dir.name.split(sep='_')[1]+'_'+img_dir.name.split(sep='_')[2].strip('.png')
            save_name =  img_dir.name.strip('.png') +'_'+str(round(X,3))+'_'+str(round(Y,3))+'.png'
            cv2.imwrite(save_path + save_name, im)
            print('Save name   :   ', save_name)


if __name__ == '__main__':
    segmented_rgb_detect(config.RGB_DIR_T,config.ADDITIONAL_SEGMENTED)
