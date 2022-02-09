import pathlib
import json
from pathlib import Path
#import rosbag
import re
import config as cfg
import tqdm
import scipy
import tensorflow_probability as tfp
import cv2
import os
from natsort import natsorted
import numpy.matlib as mat
import numpy as np
from plotting_functions import plot_traj_distribution_1_joint,plot_weights_distributions_1_joint
from ProMP import ProMP,ProMPTuner

def is_pos_def(x,tol=0):
    return np.all(np.linalg.eigvals(x) > tol)


def bag_to_json(datset_dir_for_single_config,json_path):
    '''
    Function to convert .bag files to .json files
    :param datset_dir_for_single_config: directory of the .bag files
    :param json_path: directory in which the .json images are going to be stored
    '''
    # Create output directory
    Path(json_path).mkdir(exist_ok=True, parents=True)

    for idx, bag_file in enumerate(pathlib.Path(datset_dir_for_single_config).rglob('*.bag')):

        bag = rosbag.Bag(bag_file.as_posix())
        json_file = json_path+'experiment_' + str(idx) + ".json"
        json_data = dict()
        json_data['joint_position'] = []
        json_data['joint_speed'] = []
        json_data['joint_torque'] = []
        json_data['time'] = []

        for topic, msg, t in bag.read_messages(topics=["/joint_states"]):
            json_data['time'].append(msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9)
            json_data['joint_position'].append(msg.position)
            json_data['joint_speed'].append(msg.velocity)
            json_data['joint_torque'].append(msg.effort)
        # subtract first time sample for all
        json_data['time'] = [sample - json_data['time'][0] for sample in json_data['time']]

        with open(json_file , 'w') as write_file:
          json.dump(json_data, write_file, ensure_ascii=False, indent=4)
        print('Saved as JSON file')

def PRoMPs_extraction(N_BASIS, N_DOF, N_T, json_files_path, save_file_path,joint_n = 1 ):
    n_t = N_T
    n_joints = N_DOF
    promp = ProMP(N_BASIS, N_DOF, N_T)
    config = int(Path(json_files_path).parts[-1])
    json_files = [pos_json for pos_json in os.listdir(json_files_path) if pos_json.endswith('.json')]
    json_files = natsorted(json_files)
    all_weights = []
    for index, js in enumerate(json_files):
        with open(os.path.join(json_files_path, js)) as json_file:
            json_text = json.load(json_file)
            if 'joint_position' in json_text.keys():
                joint_pos_key = 'joint_position'
            elif 'joint_pos' in json_text.keys():
                joint_pos_key = 'joint_pos'
            else:
                raise KeyError(f"Joint position not found in the trajectory file {js}")
            joint_pos = json_text[joint_pos_key]
            joint_matrix = np.vstack(np.array(joint_pos)[:,  4+joint_n])
            num_samples = joint_matrix.shape[0]
            phi = promp.basis_func_gauss_local(num_samples)  # (150,8)
            weights = np.transpose(mat.matmul(np.linalg.pinv(phi), joint_matrix))
            all_weights.append(np.hstack((weights)))
    all_weights = np.asarray(all_weights)  # (10 ,56)
    num_samples = all_weights.shape[0]
    t = np.empty(shape=(n_t, n_joints, num_samples), dtype='float64')
    for i in range(num_samples):
        t[:, :, i] = promp.trajectory_from_weights(all_weights[i, :], vector_output=False)  # (150, 7, 10)
    MEAN_TRAJ = np.mean(t, axis=-1, dtype='float64')  # (150,1)
    t = np.empty(shape=(n_t * n_joints, num_samples), dtype='float64')  # (1050)
    for i in range(num_samples):
        t[:, i] = promp.trajectory_from_weights(all_weights[i, :], vector_output=True)  # (1050,34)
    COV_TRAJ = np.cov(t, dtype='float64')
    # COV_TRAJ = COV_TRAJ + 1e-15*np.identity(1050)
    STD_TRAJ = promp.get_std_from_covariance(COV_TRAJ)
    STD_TRAJ = np.reshape(STD_TRAJ, (n_t, -1), order='F')  # (1050)
    # print('The trajectory covariance matrix is positive definite?   ', is_pos_def(COV_TRAJ))

    # ProMPs
    MEAN_WEIGHTS = promp.get_mean_from_weights(all_weights).astype('float64')  # (56,)
    all_weights = np.transpose(all_weights)  # (56,34)
    COV_WEIGHTS = promp.get_cov_from_weights(all_weights)  # (56,56)
    # COV_WEIGHTS = COV_WEIGHTS+ 1e-10*np.identity(56)
    COV_WEIGHTS = COV_WEIGHTS.astype('float64')
    STD_WEIGHTS = promp.get_std_from_covariance(COV_WEIGHTS)
    print('The weights covariance matrix is positive definite?   ', is_pos_def(COV_WEIGHTS))
    MEAN_TRAJ_PROMP = promp.trajectory_from_weights(MEAN_WEIGHTS, vector_output=False)  # (150, 7)
    COV_TRAJ_PROMP = promp.get_traj_cov(COV_WEIGHTS).astype('float64')  # (150, 150)
    # COV_TRAJ_PROMP = COV_TRAJ_PROMP + 1e-11 * np.identity(1050)
    COV_TRAJ_PROMP = COV_TRAJ_PROMP.astype('float64')
    STD_TRAJ_PROMP = promp.get_std_from_covariance(COV_TRAJ_PROMP)  # (1050)
    STD_TRAJ_PROMP = np.reshape(STD_TRAJ_PROMP, (n_t, -1), order='F')
    # print('The ProMPs traj covariance matrix is positive definite?   ', is_pos_def(COV_TRAJ_PROMP))

    plot_traj_distribution_1_joint(save_path=save_file_path, config=config, mean_traj_1=MEAN_TRAJ_PROMP,
                                   traj_std_1=STD_TRAJ_PROMP, mean_traj_2=MEAN_TRAJ, traj_std_2=STD_TRAJ, show=False,
                                   save=True)
    # plot_weights_distributions_1_joint(save_file_path=save_file_path,mean_weights_1=MEAN_WEIGHTS,std_weights_1=STD_WEIGHTS,n_func=8,mean_weights_2=np.zeros(shape=(8,)), std_weights_2 = np.zeros(shape=(8,)),show = False,save = False)
    return MEAN_TRAJ_PROMP, COV_TRAJ_PROMP, MEAN_WEIGHTS, COV_WEIGHTS

def rename_images_rgb(rgb_dir,save_dir):
    for i in range(25):
        for idx, img_dir in enumerate(pathlib.Path(rgb_dir + str(i) + '/').rglob('*.png')):
            if img_dir.name.split(sep="_")[2] == 't':
                print(i,'   :   ',img_dir.name)
                im = cv2.imread(img_dir.as_posix())

                ''' Configuration definintion '''
                if i <= 4:
                    config = 1
                if i > 4 and i <= 9:
                    config = 2
                if i > 9 and i <= 14:
                    config = 3
                if i > 14 and i <= 19:
                    config = 4
                if i > 19 and i <= 24:
                    config = 5

                ''' Berry definintion '''
                if i == 0 or i==5 or i==10 or i==15 or i==20:
                    straw = 1
                if i == 1 or i==6 or i==11 or i==16 or i==21:
                    straw = 2
                if i == 2 or i==7 or i==12 or i==17 or i==22:
                    straw = 3
                if i == 3 or i==8 or i==13 or i==18 or i==23:
                    straw = 4
                if i == 4 or i==9 or i==14 or i==19 or i==24:
                    straw = 5

                name_segmented_image = save_dir + str(i) +'_conf'+str(config)+'_'+ (img_dir).name.strip('.png').strip('rgb_img') + '_berry'+str(int(straw))+'.png'
                cv2.imwrite(name_segmented_image, im)




def load_bb_and_save(img_dir_, bbox_dir,save_path):
    for i in range(9,10):
        for idx, img_dir in tqdm(enumerate(pathlib.Path(img_dir_+'/'+ str(i)+'/').rglob('*.png'))):
            bbox_d = bbox_dir+ str(i)+'/'+ (img_dir).name.strip('.png') + '.txt'
            if os.path.exists(bbox_d):
                with open(bbox_d) as f:
                    lines = f.readlines()
                    for berry in range(1):
                        img = cv2.imread(str(img_dir))
                        bbox = lines[berry].split(' ')
                        l,x, y, width, height = bbox
                        x1,y1,x2,y2,X_center,Y_center= yolobbox2bbox(float(x),float(y),float(width),float(height),640,480)
                        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0),-1)
                        #cv2.imshow(str(berry), img)
                        #cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                        ''' Configuration definintion '''
                        if i<=4:
                            config = 1
                        if i>4 and i<=9:
                            config = 2
                        if i>9 and i<=14:
                            config = 3
                        if i>14 and i<=19:
                            config = 4
                        if i > 19 and i <= 24:
                            config = 5


                        Path(save_path).mkdir(exist_ok=True, parents=True)
                        save_name = save_path+ str(i) +'_conf'+str(config)+'_'+ (img_dir).name.strip('.png').strip('rgb_img') + '_berry'+str(int(l)+1)+'_'+str(round(X_center,3))+'_'+str(round(Y_center,3))+'.png'
                        cv2.imwrite(save_name,img)
                        print(save_name+'  SAVED')



if __name__ == '__main__':
    '''
    1. Transform .bag to.json files
    '''
    # for i in range(25):
    #  datset_dir = cfg.COLLECTION_DIR + str(i) +'/'
    #  save_json_path= cfg.TRAJ_DIR + str(i) +'/'
    #  bag_to_json(datset_dir, save_json_path)


    '''
    2. Extract ProMPs
    '''
    # N_BASIS = 8
    # N_DOF = 1
    # N_T = 150
    # promp = ProMP(N_BASIS, N_DOF, N_T)
    # for j in range(1,8):
    #     for i in range(25):
    #         json_file_path = cfg.TRAJ_DIR + str(i) +'/'
    #         img_save_file_path = cfg.PLOT_DATA_PATH +'J'+ str(j) +'/'
    #         save_annotation_path = cfg.ANNOTATION_PATH +'J'+ str(j) +'/'
    #         Path(save_annotation_path).mkdir(exist_ok=True, parents=True)
    #
    #         MEAN_TRAJ, COV_TRAJ, MEAN_WEIGHTS, COV_WEIGHTS = PRoMPs_extraction(N_BASIS, N_DOF, N_T,
    #                                                                            json_files_path=json_file_path,
    #                                                                            save_file_path=img_save_file_path,joint_n = j )
    #
    #         # Build a lower triangular matrix, with the diagonal values of log(D) and the lower values of L
    #         L, D, _ = scipy.linalg.ldl(COV_WEIGHTS)
    #         d = np.diag(D)
    #         L_new = L
    #         L_new[np.diag_indices(8)] = d
    #         tril_elements = tfp.math.fill_triangular_inverse(L_new)
    #
    #         # SAVE ANNOTATION
    #         config = int(Path(json_file_path).parts[-1])
    #         annotation = {}
    #         annotation["mean_weights"] = np.vstack(MEAN_WEIGHTS).tolist()
    #         annotation["L"] = tril_elements.numpy().tolist()
    #         annotation["configuration"] = config
    #         dump_file_name = str(config) + '.json'
    #         dump_file_path = save_annotation_path + dump_file_name
    #         with open(dump_file_path, 'w') as f:
    #             json.dump(annotation, f)

    rename_images_rgb(cfg.RGB_DIR ,cfg.RGB_DIR_T)
