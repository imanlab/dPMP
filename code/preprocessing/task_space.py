from kinematics_UPH import joint_to_cartesian_ee_position
from ProMP import ProMP,ProMPTuner
from pathlib import Path
import tensorflow_probability as tfp
import scipy
from natsort import natsorted
import json
import os
import numpy as np
import numpy.matlib as mat
import config as cfg
from plotting_functions import plot_traj_distribution_1_joint
def is_pos_def(x,tol=0):
    return np.all(np.linalg.eigvals(x) > tol)

def PRoMPs_extraction(N_BASIS, N_DOF, N_T, json_files_path, save_file_path):
    n_t = N_T
    n_joints = N_DOF
    promp = ProMP(N_BASIS, N_DOF, N_T)
    config = int(Path(json_files_path).parts[-1])
    json_files = [pos_json for pos_json in os.listdir(json_files_path) if pos_json.endswith('.json')]
    json_files = natsorted(json_files)
    all_weights_1 = []
    all_weights_2 = []
    all_weights_3 = []
    all_weights_4 = []
    all_weights_5 = []
    all_weights_6 = []
    all_weights_7 = []

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

            joint_matrix_1 = np.vstack(np.array(joint_pos)[:,  4+1])
            num_samples_1 = joint_matrix_1.shape[0]
            phi_1 = promp.basis_func_gauss_local(num_samples_1)  # (150,8)
            weights_1 = np.transpose(mat.matmul(np.linalg.pinv(phi_1), joint_matrix_1))
            all_weights_1.append(np.hstack((weights_1)))

            joint_matrix_2 = np.vstack(np.array(joint_pos)[:,  4+2])
            num_samples_2= joint_matrix_2.shape[0]
            phi_2 = promp.basis_func_gauss_local(num_samples_2)  # (150,8)
            weights_2 = np.transpose(mat.matmul(np.linalg.pinv(phi_2), joint_matrix_2))
            all_weights_2.append(np.hstack((weights_2)))

            joint_matrix_3 = np.vstack(np.array(joint_pos)[:,  4+3])
            num_samples_3 = joint_matrix_3.shape[0]
            phi_3 = promp.basis_func_gauss_local(num_samples_3)  # (150,8)
            weights_3 = np.transpose(mat.matmul(np.linalg.pinv(phi_3), joint_matrix_3))
            all_weights_3.append(np.hstack((weights_3)))

            joint_matrix_4 = np.vstack(np.array(joint_pos)[:,  4+4])
            num_samples_4 = joint_matrix_4.shape[0]
            phi_4 = promp.basis_func_gauss_local(num_samples_4)  # (150,8)
            weights_4 = np.transpose(mat.matmul(np.linalg.pinv(phi_4), joint_matrix_4))
            all_weights_4.append(np.hstack((weights_4)))

            joint_matrix_5 = np.vstack(np.array(joint_pos)[:,  4+5])
            num_samples_5 = joint_matrix_5.shape[0]
            phi_5 = promp.basis_func_gauss_local(num_samples_5)  # (150,8)
            weights_5 = np.transpose(mat.matmul(np.linalg.pinv(phi_5), joint_matrix_5))
            all_weights_5.append(np.hstack((weights_5)))

            joint_matrix_6 = np.vstack(np.array(joint_pos)[:,  4+6])
            num_samples_6  = joint_matrix_6.shape[0]
            phi_6 = promp.basis_func_gauss_local(num_samples_6)  # (150,8)
            weights_6 = np.transpose(mat.matmul(np.linalg.pinv(phi_6), joint_matrix_6))
            all_weights_6.append(np.hstack((weights_6)))

            joint_matrix_7 = np.vstack(np.array(joint_pos)[:,  4+7])
            num_samples_7 = joint_matrix_7.shape[0]
            phi_7 = promp.basis_func_gauss_local(num_samples_7)  # (150,8)
            weights_7 = np.transpose(mat.matmul(np.linalg.pinv(phi_7), joint_matrix_7))
            all_weights_7.append(np.hstack((weights_7)))

    all_weights_1 = np.asarray(all_weights_1)  # (10 ,8)
    num_samples_1 = all_weights_1.shape[0]
    t_1 = np.empty(shape=(n_t, num_samples_1), dtype='float64')

    all_weights_2 = np.asarray(all_weights_2)  # (10 ,8)
    num_samples_2 = all_weights_2.shape[0]
    t_2 = np.empty(shape=(n_t, num_samples_2), dtype='float64')

    all_weights_3 = np.asarray(all_weights_3)  # (10 ,8)
    num_samples_3 = all_weights_3.shape[0]
    t_3 = np.empty(shape=(n_t, num_samples_3), dtype='float64')

    all_weights_4 = np.asarray(all_weights_4)  # (10 ,8)
    num_samples_4 = all_weights_4.shape[0]
    t_4 = np.empty(shape=(n_t, num_samples_4), dtype='float64')

    all_weights_5 = np.asarray(all_weights_5)  # (10 ,8)
    num_samples_5 = all_weights_5.shape[0]
    t_5 = np.empty(shape=(n_t, num_samples_5), dtype='float64')

    all_weights_6 = np.asarray(all_weights_6)  # (10 ,8)
    num_samples_6 = all_weights_6.shape[0]
    t_6 = np.empty(shape=(n_t, num_samples_6), dtype='float64')

    all_weights_7 = np.asarray(all_weights_7)  # (10 ,8)
    num_samples_7 = all_weights_7.shape[0]
    t_7 = np.empty(shape=(n_t, num_samples_7), dtype='float64')

    for i in range(num_samples_1):
        t_1[:,  i] = promp.trajectory_from_weights(all_weights_1[i, :], vector_output=False).squeeze() #(150, 11)

    for i in range(num_samples_2):
        t_2[:,  i] = promp.trajectory_from_weights(all_weights_2[i, :], vector_output=False).squeeze() #(150, 11)

    for i in range(num_samples_3):
        t_3[:,  i] = promp.trajectory_from_weights(all_weights_3[i, :], vector_output=False).squeeze() #(150, 11)

    for i in range(num_samples_4):
        t_4[:,  i] = promp.trajectory_from_weights(all_weights_4[i, :], vector_output=False).squeeze() #(150, 11)

    for i in range(num_samples_5):
        t_5[:,  i] = promp.trajectory_from_weights(all_weights_5[i, :], vector_output=False).squeeze() #(150, 11)

    for i in range(num_samples_6):
        t_6[:,  i] = promp.trajectory_from_weights(all_weights_6[i, :], vector_output=False).squeeze() #(150, 11)

    for i in range(num_samples_7):
        t_7[:,  i] = promp.trajectory_from_weights(all_weights_7[i, :], vector_output=False).squeeze() #(150, 11)

    ''' task space '''

    q = np.empty(shape=(7,n_t,num_samples_1), dtype='float64')
    q[0,:,:] = t_1
    q[1,:,:] = t_2
    q[2,:,:] = t_3
    q[3,:,:] = t_4
    q[4,:,:] = t_5
    q[5,:,:] = t_6
    q[6,:,:] = t_7

    x = np.empty(shape=(n_t,num_samples_1), dtype='float64')
    y = np.empty(shape=(n_t,num_samples_1), dtype='float64')
    z = np.empty(shape=(n_t,num_samples_1), dtype='float64')
    q1 = np.empty(shape=(n_t,num_samples_1), dtype='float64')
    q2 = np.empty(shape=(n_t,num_samples_1), dtype='float64')
    q3 = np.empty(shape=(n_t,num_samples_1), dtype='float64')
    q4 = np.empty(shape=(n_t,num_samples_1), dtype='float64')

    for time in range(n_t):
        for traj in range(num_samples_1):
            x[time,traj], y[time,traj], z[time,traj], q1[time,traj], q2[time,traj], q3[time,traj], q4[time,traj] = joint_to_cartesian_ee_position(q[:,time,traj])

    x_mean = np.mean(x, axis=-1, dtype='float64')
    y_mean = np.mean(y, axis=-1, dtype='float64')
    z_mean =np.mean(z, axis=-1, dtype='float64')
    q1_mean = np.mean(q1, axis=-1, dtype='float64')
    q2_mean = np.mean(q2, axis=-1, dtype='float64')
    q3_mean =np.mean(q3, axis=-1, dtype='float64')
    q4_mean = np.mean(q4, axis=-1, dtype='float64')

    x_cov = np.cov(x.astype('float64'), dtype='float64')
    y_cov = np.cov(y.astype('float64'), dtype='float64')
    z_cov =np.cov(z.astype('float64'), dtype='float64')
    q1_cov = np.cov(q1.astype('float64'), dtype='float64')
    q2_cov = np.cov(q2.astype('float64'),  dtype='float64')
    q3_cov =np.cov(q3.astype('float64'),  dtype='float64')
    q4_cov = np.cov(q4.astype('float64'), dtype='float64')

    x_std = promp.get_std_from_covariance(x_cov)
    y_std = promp.get_std_from_covariance(y_cov)
    z_std = promp.get_std_from_covariance(z_cov)
    q1_std = promp.get_std_from_covariance(q1_cov)
    q2_std = promp.get_std_from_covariance(q2_cov)
    q3_std = promp.get_std_from_covariance(q3_cov)
    q4_std = promp.get_std_from_covariance(q4_cov)


    weights_x_tot = []
    weights_y_tot = []
    weights_z_tot = []
    weights_q1_tot = []
    weights_q2_tot = []
    weights_q3_tot = []
    weights_q4_tot = []

    for sample in range(num_samples_1):
        phi_x = promp.basis_func_gauss_local(n_t)  # (150,8)
        weights_x = np.transpose(mat.matmul(np.linalg.pinv(phi_x), x[:,sample]))
        weights_x_tot.append(np.hstack((weights_x)))

        phi_y = promp.basis_func_gauss_local(n_t)  # (150,8)
        weights_y = np.transpose(mat.matmul(np.linalg.pinv(phi_y), y[:,sample]))
        weights_y_tot.append(np.hstack((weights_y)))

        phi_z = promp.basis_func_gauss_local(n_t)  # (150,8)
        weights_z = np.transpose(mat.matmul(np.linalg.pinv(phi_z), z[:,sample]))
        weights_z_tot.append(np.hstack((weights_z)))

        phi_q1 = promp.basis_func_gauss_local(n_t)  # (150,8)
        weights_q1 = np.transpose(mat.matmul(np.linalg.pinv(phi_q1), q1[:,sample]))
        weights_q1_tot.append(np.hstack((weights_q1)))

        phi_q2 = promp.basis_func_gauss_local(n_t)  # (150,8)
        weights_q2 = np.transpose(mat.matmul(np.linalg.pinv(phi_q2), q2[:,sample]))
        weights_q2_tot.append(np.hstack((weights_q2)))

        phi_q3 = promp.basis_func_gauss_local(n_t)  # (150,8)
        weights_q3 = np.transpose(mat.matmul(np.linalg.pinv(phi_q3), q3[:,sample]))
        weights_q3_tot.append(np.hstack((weights_q3)))

        phi_q4 = promp.basis_func_gauss_local(n_t)  # (150,8)
        weights_q4 = np.transpose(mat.matmul(np.linalg.pinv(phi_q4), q4[:,sample]))
        weights_q4_tot.append(np.hstack((weights_q4)))

    weights_x_tot = np.asarray(weights_x_tot)
    weights_y_tot = np.asarray(weights_y_tot)
    weights_z_tot = np.asarray(weights_z_tot)
    weights_q1_tot = np.asarray(weights_q1_tot)
    weights_q2_tot = np.asarray(weights_q2_tot)
    weights_q3_tot = np.asarray(weights_q3_tot)
    weights_q4_tot = np.asarray(weights_q4_tot)


    x_reconstructed = np.empty(shape=(n_t, num_samples_1), dtype='float64')
    y_reconstructed = np.empty(shape=(n_t, num_samples_1), dtype='float64')
    z_reconstructed = np.empty(shape=(n_t, num_samples_1), dtype='float64')
    q1_reconstructed = np.empty(shape=(n_t, num_samples_1), dtype='float64')
    q2_reconstructed = np.empty(shape=(n_t, num_samples_1), dtype='float64')
    q3_reconstructed = np.empty(shape=(n_t, num_samples_1), dtype='float64')
    q4_reconstructed = np.empty(shape=(n_t, num_samples_1), dtype='float64')


    for i in range(num_samples_1):
        x_reconstructed[:,  i] = promp.trajectory_from_weights(weights_x_tot[i, :], vector_output=False).squeeze() #(150, 11)
        y_reconstructed[:,  i] = promp.trajectory_from_weights(weights_y_tot[i, :], vector_output=False).squeeze() #(150, 11)
        z_reconstructed[:,  i] = promp.trajectory_from_weights(weights_z_tot[i, :], vector_output=False).squeeze() #(150, 11)
        q1_reconstructed[:,  i] = promp.trajectory_from_weights(weights_q1_tot[i, :], vector_output=False).squeeze() #(150, 11)
        q2_reconstructed[:,  i] = promp.trajectory_from_weights(weights_q2_tot[i, :], vector_output=False).squeeze() #(150, 11)
        q3_reconstructed[:,  i] = promp.trajectory_from_weights(weights_q3_tot[i, :], vector_output=False).squeeze() #(150, 11)
        q4_reconstructed[:,  i] = promp.trajectory_from_weights(weights_q4_tot[i, :], vector_output=False).squeeze() #(150, 11)



    MEAN_TRAJ_X = np.mean(x_reconstructed, axis=-1, dtype='float64')  # (150,1)
    MEAN_TRAJ_Y = np.mean(y_reconstructed, axis=-1, dtype='float64')  # (150,1)
    MEAN_TRAJ_Z = np.mean(z_reconstructed, axis=-1, dtype='float64')  # (150,1)
    MEAN_TRAJ_Q1 = np.mean(q1_reconstructed, axis=-1, dtype='float64')  # (150,1)
    MEAN_TRAJ_Q2 = np.mean(q2_reconstructed, axis=-1, dtype='float64')  # (150,1)
    MEAN_TRAJ_Q3 = np.mean(q3_reconstructed, axis=-1, dtype='float64')  # (150,1)
    MEAN_TRAJ_Q4 = np.mean(q4_reconstructed, axis=-1, dtype='float64')  # (150,1)

    for i in range(num_samples_1):
         x_reconstructed.reshape((-1,)).astype('float64')
         y_reconstructed.reshape((-1,)).astype('float64')
         z_reconstructed.reshape((-1,)).astype('float64')
         q1_reconstructed.reshape((-1,)).astype('float64')
         q2_reconstructed.reshape((-1,)).astype('float64')
         q3_reconstructed.reshape((-1,)).astype('float64')
         q4_reconstructed.reshape((-1,)).astype('float64')

    COV_TRAJ_X = np.cov(x_reconstructed,  dtype='float64') #+ 1e-15*np.identity(150)  # (150,150)
    COV_TRAJ_Y = np.cov(y_reconstructed,  dtype='float64') #+ 1e-15*np.identity(150) # (150,150)
    COV_TRAJ_Z = np.cov(z_reconstructed,  dtype='float64') #+ 1e-15*np.identity(150) # (150,150)
    COV_TRAJ_Q1 = np.cov(q1_reconstructed, dtype='float64') #+ 1e-15*np.identity(150) # (150,150)
    COV_TRAJ_Q2 = np.cov(q2_reconstructed, dtype='float64') #+ 1e-15*np.identity(150) # (150,150)
    COV_TRAJ_Q3 = np.cov(q3_reconstructed, dtype='float64') #+ 1e-15*np.identity(150) # (150,150)
    COV_TRAJ_Q4 = np.cov(q4_reconstructed, dtype='float64') #+ 1e-15*np.identity(150) # (150,150)

    STD_TRAJ_Q1 = np.reshape(promp.get_std_from_covariance(COV_TRAJ_Q1), (n_t, -1), order='F')
    STD_TRAJ_Q2 = np.reshape(promp.get_std_from_covariance(COV_TRAJ_Q2), (n_t, -1), order='F')
    STD_TRAJ_Q3 = np.reshape(promp.get_std_from_covariance(COV_TRAJ_Q3), (n_t, -1), order='F')
    STD_TRAJ_Q4 = np.reshape(promp.get_std_from_covariance(COV_TRAJ_Q4), (n_t, -1), order='F')

    STD_TRAJ_X = np.reshape(promp.get_std_from_covariance(COV_TRAJ_X), (n_t, -1), order='F')
    STD_TRAJ_Y = np.reshape(promp.get_std_from_covariance(COV_TRAJ_Y), (n_t, -1), order='F')
    STD_TRAJ_Z = np.reshape(promp.get_std_from_covariance(COV_TRAJ_Z), (n_t, -1), order='F')


    # ProMPs
    MEAN_WEIGHTS_X = promp.get_mean_from_weights(weights_x_tot).astype('float64')  # (56,)
    COV_WEIGHTS_X = promp.get_cov_from_weights(np.transpose(weights_x_tot) ).astype('float64') # (56,56)
    STD_WEIGHTS_X = promp.get_std_from_covariance(COV_WEIGHTS_X)

    MEAN_WEIGHTS_Y = promp.get_mean_from_weights(weights_y_tot).astype('float64')  # (56,)
    COV_WEIGHTS_Y = promp.get_cov_from_weights(np.transpose(weights_y_tot) ).astype('float64') # (56,56)
    STD_WEIGHTS_Y = promp.get_std_from_covariance(COV_WEIGHTS_Y)

    MEAN_WEIGHTS_Z = promp.get_mean_from_weights(weights_z_tot).astype('float64')  # (56,)
    COV_WEIGHTS_Z = promp.get_cov_from_weights(np.transpose(weights_z_tot) ).astype('float64') # (56,56)
    STD_WEIGHTS_Z = promp.get_std_from_covariance(COV_WEIGHTS_Z)

    MEAN_WEIGHTS_Q1 = promp.get_mean_from_weights(weights_q1_tot).astype('float64')  # (56,)
    COV_WEIGHTS_Q1= promp.get_cov_from_weights(np.transpose(weights_q1_tot) ).astype('float64') # (56,56)
    STD_WEIGHTS_Q1 = promp.get_std_from_covariance(COV_WEIGHTS_Q1)

    MEAN_WEIGHTS_Q2 = promp.get_mean_from_weights(weights_q2_tot).astype('float64')  # (56,)
    COV_WEIGHTS_Q2 = promp.get_cov_from_weights(np.transpose(weights_q2_tot) ).astype('float64') # (56,56)
    STD_WEIGHTS_Q2 = promp.get_std_from_covariance(COV_WEIGHTS_Q2)

    MEAN_WEIGHTS_Q3 = promp.get_mean_from_weights(weights_q3_tot).astype('float64')  # (56,)
    COV_WEIGHTS_Q3 = promp.get_cov_from_weights(np.transpose(weights_q3_tot)).astype('float64')  # (56,56)
    STD_WEIGHTS_Q3 = promp.get_std_from_covariance(COV_WEIGHTS_Q3)

    MEAN_WEIGHTS_Q4 = promp.get_mean_from_weights(weights_q4_tot).astype('float64')  # (56,)
    COV_WEIGHTS_Q4 = promp.get_cov_from_weights(np.transpose(weights_q4_tot)).astype('float64')  # (56,56)
    STD_WEIGHTS_Q4 = promp.get_std_from_covariance(COV_WEIGHTS_Q4)


    plot_traj_distribution_1_joint(save_path=save_file_path, config=config, xyz='x',mean_traj_1=x_mean,
                                     traj_std_1=x_std,mean_traj_2=MEAN_TRAJ_X,
                                     traj_std_2=STD_TRAJ_X.squeeze(), show=False,
                                     save=True)

    plot_traj_distribution_1_joint(save_path=save_file_path, config=config, xyz='y',mean_traj_1=y_mean,
                                     traj_std_1=y_std,mean_traj_2=MEAN_TRAJ_Y,
                                     traj_std_2=STD_TRAJ_Y.squeeze(), show=False,
                                     save=True)

    plot_traj_distribution_1_joint(save_path=save_file_path, config=config, xyz='z',mean_traj_1=z_mean,
                                     traj_std_1=z_std,mean_traj_2=MEAN_TRAJ_Z,
                                     traj_std_2=STD_TRAJ_Z.squeeze(), show=False,
                                     save=True)

    plot_traj_distribution_1_joint(save_path=save_file_path, config=config,xyz='q1', mean_traj_1=q1_mean,
                                     traj_std_1=q1_std,mean_traj_2=MEAN_TRAJ_Q1,
                                     traj_std_2=STD_TRAJ_Q1.squeeze(), show=False,
                                     save=True)

    plot_traj_distribution_1_joint(save_path=save_file_path, config=config,xyz='q2', mean_traj_1=q2_mean,
                                     traj_std_1=q2_std,mean_traj_2=MEAN_TRAJ_Q2,
                                     traj_std_2=STD_TRAJ_Q2.squeeze(), show=False,
                                     save=True)

    plot_traj_distribution_1_joint(save_path=save_file_path, config=config,xyz='q3', mean_traj_1=q3_mean,
                                   traj_std_1=q3_std, mean_traj_2=MEAN_TRAJ_Q3,
                                   traj_std_2=STD_TRAJ_Q3.squeeze(), show=False,
                                   save=True)

    plot_traj_distribution_1_joint(save_path=save_file_path, config=config,xyz='q4', mean_traj_1=q4_mean,
                                   traj_std_1=q4_std, mean_traj_2=MEAN_TRAJ_Q4,
                                   traj_std_2=STD_TRAJ_Q4.squeeze(), show=False,
                                   save=True)


    return MEAN_WEIGHTS_X,MEAN_WEIGHTS_Y,MEAN_WEIGHTS_Z,MEAN_WEIGHTS_Q1,MEAN_WEIGHTS_Q2,MEAN_WEIGHTS_Q3,MEAN_WEIGHTS_Q4,COV_WEIGHTS_X,COV_WEIGHTS_Y,COV_WEIGHTS_Z,COV_WEIGHTS_Q1,COV_WEIGHTS_Q2,COV_WEIGHTS_Q3,COV_WEIGHTS_Q4

if __name__ == '__main__':
    '''
    2. Extract ProMPs
    '''
    N_BASIS = 8
    N_DOF = 1
    N_T = 150
    promp = ProMP(N_BASIS, N_DOF, N_T)
    for i in range(25):
            json_file_path = cfg.TRAJ_DIR + str(i) +'/'
            img_save_file_path = cfg.PLOT_TASK
            save_annotation_path_x = cfg.TASK_SPACE_ANNOTATION +'x' +'/'
            save_annotation_path_y = cfg.TASK_SPACE_ANNOTATION +'y' +'/'
            save_annotation_path_z = cfg.TASK_SPACE_ANNOTATION +'z' +'/'
            save_annotation_path_q1 = cfg.TASK_SPACE_ANNOTATION +'q1' +'/'
            save_annotation_path_q2 = cfg.TASK_SPACE_ANNOTATION +'q2' +'/'
            save_annotation_path_q3 = cfg.TASK_SPACE_ANNOTATION +'q3' +'/'
            save_annotation_path_q4 = cfg.TASK_SPACE_ANNOTATION +'q4' +'/'

            Path(save_annotation_path_x).mkdir(exist_ok=True, parents=True)
            Path(save_annotation_path_y).mkdir(exist_ok=True, parents=True)
            Path(save_annotation_path_z).mkdir(exist_ok=True, parents=True)
            Path(save_annotation_path_q1).mkdir(exist_ok=True, parents=True)
            Path(save_annotation_path_q2).mkdir(exist_ok=True, parents=True)
            Path(save_annotation_path_q3).mkdir(exist_ok=True, parents=True)
            Path(save_annotation_path_q4).mkdir(exist_ok=True, parents=True)



            MEAN_WEIGHTS_X,MEAN_WEIGHTS_Y,MEAN_WEIGHTS_Z,MEAN_WEIGHTS_Q1,MEAN_WEIGHTS_Q2,MEAN_WEIGHTS_Q3,MEAN_WEIGHTS_Q4,COV_WEIGHTS_X,COV_WEIGHTS_Y,COV_WEIGHTS_Z,COV_WEIGHTS_Q1,COV_WEIGHTS_Q2,COV_WEIGHTS_Q3,COV_WEIGHTS_Q4 = PRoMPs_extraction(N_BASIS, N_DOF, N_T,json_files_path=json_file_path,save_file_path=img_save_file_path )

            #Build a lower triangular matrix, with the diagonal values of log(D) and the lower values of L
            L_x, D_x, _ = scipy.linalg.ldl(COV_WEIGHTS_X)
            d_x = np.diag(D_x)
            L_new_x = L_x
            L_new_x[np.diag_indices(8)] = d_x
            tril_elements_x = tfp.math.fill_triangular_inverse(L_new_x)

            L_y, D_y, _ = scipy.linalg.ldl(COV_WEIGHTS_Y)
            d_y = np.diag(D_y)
            L_new_y= L_y
            L_new_y[np.diag_indices(8)] = d_y
            tril_elements_y = tfp.math.fill_triangular_inverse(L_new_y)

            L_z, D_z, _ = scipy.linalg.ldl(COV_WEIGHTS_Z)
            d_z = np.diag(D_z)
            L_new_z = L_z
            L_new_z[np.diag_indices(8)] = d_z
            tril_elements_z = tfp.math.fill_triangular_inverse(L_new_z)

            L_q1, D_q1, _ = scipy.linalg.ldl(COV_WEIGHTS_Q1)
            d_q1 = np.diag(D_q1)
            L_new_q1 = L_q1
            L_new_q1[np.diag_indices(8)] = d_q1
            tril_elements_q1 = tfp.math.fill_triangular_inverse(L_new_q1)

            L_q2, D_q2, _ = scipy.linalg.ldl(COV_WEIGHTS_Q2)
            d_q2 = np.diag(D_q2)
            L_new_q2 = L_q2
            L_new_q2[np.diag_indices(8)] = d_q2
            tril_elements_q2 = tfp.math.fill_triangular_inverse(L_new_q2)

            L_q3, D_q3, _ = scipy.linalg.ldl(COV_WEIGHTS_Q3)
            d_q3= np.diag(D_q3)
            L_new_q3 = L_q3
            L_new_q3[np.diag_indices(8)] = d_q3
            tril_elements_q3 = tfp.math.fill_triangular_inverse(L_new_q3)

            L_q4, D_q4, _ = scipy.linalg.ldl(COV_WEIGHTS_Q4)
            d_q4 = np.diag(D_q4)
            L_new_q4 = L_q4
            L_new_q4[np.diag_indices(8)] = d_q4
            tril_elements_q4 = tfp.math.fill_triangular_inverse(L_new_q4)

            # SAVE ANNOTATION
            config = int(Path(json_file_path).parts[-1])
            dump_file_name = str(config) + '.json'

            annotation_x = {}
            annotation_x["mean_weights"] = np.vstack(MEAN_WEIGHTS_X).tolist()
            annotation_x["L"] = tril_elements_x.numpy().tolist()
            annotation_x["configuration"] = config
            dump_file_path_x = save_annotation_path_x + dump_file_name
            with open(dump_file_path_x, 'w') as f_x:
                json.dump(annotation_x, f_x)

            annotation_y = {}
            annotation_y["mean_weights"] = np.vstack(MEAN_WEIGHTS_Y).tolist()
            annotation_y["L"] = tril_elements_y.numpy().tolist()
            annotation_y["configuration"] = config
            dump_file_path_y = save_annotation_path_y + dump_file_name
            with open(dump_file_path_y, 'w') as f_y:
                json.dump(annotation_y, f_y)

            annotation_z = {}
            annotation_z["mean_weights"] = np.vstack(MEAN_WEIGHTS_Z).tolist()
            annotation_z["L"] = tril_elements_z.numpy().tolist()
            annotation_z["configuration"] = config
            dump_file_path_z = save_annotation_path_z + dump_file_name
            with open(dump_file_path_z, 'w') as f_z:
                json.dump(annotation_z, f_z)

            annotation_q1 = {}
            annotation_q1["mean_weights"] = np.vstack(MEAN_WEIGHTS_Q1).tolist()
            annotation_q1["L"] = tril_elements_q1.numpy().tolist()
            annotation_q1["configuration"] = config
            dump_file_path_q1 = save_annotation_path_q1 + dump_file_name
            with open(dump_file_path_q1, 'w') as f_q1:
                json.dump(annotation_q1, f_q1)

            annotation_q2 = {}
            annotation_q2["mean_weights"] = np.vstack(MEAN_WEIGHTS_Q2).tolist()
            annotation_q2["L"] = tril_elements_q2.numpy().tolist()
            annotation_q2["configuration"] = config
            dump_file_path_q2 = save_annotation_path_q2 + dump_file_name
            with open(dump_file_path_q2, 'w') as f_q2:
                json.dump(annotation_q2, f_q2)

            annotation_q3 = {}
            annotation_q3["mean_weights"] = np.vstack(MEAN_WEIGHTS_Q3).tolist()
            annotation_q3["L"] = tril_elements_q3.numpy().tolist()
            annotation_q3["configuration"] = config
            dump_file_path_q3 = save_annotation_path_q3 + dump_file_name
            with open(dump_file_path_q3, 'w') as f_q3:
                json.dump(annotation_q3, f_q3)

            annotation_q4 = {}
            annotation_q4["mean_weights"] = np.vstack(MEAN_WEIGHTS_Q4).tolist()
            annotation_q4["L"] = tril_elements_q4.numpy().tolist()
            annotation_q4["configuration"] = config
            dump_file_path_q4 = save_annotation_path_q4 + dump_file_name
            with open(dump_file_path_q4, 'w') as f_q4:
                json.dump(annotation_q4, f_q4)
