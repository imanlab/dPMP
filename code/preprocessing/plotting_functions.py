import numpy as np
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from pathlib import Path

tfd = tfp.distributions

def plot_traj_distributions(save_path,config, mean_traj_1,traj_std_1,mean_traj_2=np.zeros(shape=(1050,)),traj_std_2=np.zeros(shape=(1050,)),show=False,save=False):
            #TRAJ 1
            traj_true_mean=mean_traj_1
            right_traj_1_true = mean_traj_1+traj_std_1
            left_traj_1_true = mean_traj_1-traj_std_1
            right_traj_3_true=mean_traj_1+3*traj_std_1
            left_traj_3_true = mean_traj_1-3*traj_std_1

            #TRAJ 2
            if  mean_traj_2.all() != 0:
                traj_pred_mean=mean_traj_2
                right_traj_1_pred = mean_traj_2+traj_std_2
                left_traj_1_pred = mean_traj_2-traj_std_2
                right_traj_3_pred =mean_traj_2+3*traj_std_2
                left_traj_3_pred = mean_traj_2-3*traj_std_2

            q1true, q2true, q3true, q4true, q5true, q6true, q7true = traj_true_mean[:, [0]], traj_true_mean[:,[1]], traj_true_mean[:, [2]], traj_true_mean[:,[3]], \
                                                                     traj_true_mean[:, [4]], traj_true_mean[:,[5]], traj_true_mean[:, [6]]
            q1righttrue, q2righttrue, q3righttrue, q4righttrue, q5righttrue, q6righttrue, q7righttrue = right_traj_3_true[:, [0]], right_traj_3_true[:,[1]], right_traj_3_true[:, [2]],\
                                                                                                        right_traj_3_true[:,[3]], right_traj_3_true[:, [4]], right_traj_3_true[:,[5]], \
                                                                                                        right_traj_3_true[:, [6]]
            q1lefttrue, q2lefttrue, q3lefttrue, q4lefttrue, q5lefttrue, q6lefttrue, q7lefttrue = left_traj_3_true[:,[0]], left_traj_3_true[:,[1]], left_traj_3_true[:,[2]], \
                                                                                                 left_traj_3_true[:,[3]], left_traj_3_true[:,[4]], left_traj_3_true[:,[5]], \
                                                                                                 left_traj_3_true[:,[6]]
            q1righttrue_1, q2righttrue_1, q3righttrue_1, q4righttrue_1, q5righttrue_1, q6righttrue_1, q7righttrue_1 = right_traj_1_true[:,[0]], right_traj_1_true[:,[1]], right_traj_1_true[:,[2]], \
                                                                                                                      right_traj_1_true[:,[ 3]], right_traj_1_true[:,[4]], right_traj_1_true[:,[5]], \
                                                                                                                      right_traj_1_true[:,[6]]
            q1lefttrue_1, q2lefttrue_1, q3lefttrue_1, q4lefttrue_1, q5lefttrue_1, q6lefttrue_1, q7lefttrue_1 = left_traj_1_true[:, [0]], left_traj_1_true[:,[1]],left_traj_1_true[:, [2]], \
                                                                                                               left_traj_1_true[:,[3]],left_traj_1_true[:, [4]], left_traj_1_true[:,[5]], \
                                                                                                               left_traj_1_true[:, [6]]

            if  mean_traj_2.all() != 0:
                q1pred, q2pred, q3pred, q4pred, q5pred, q6pred, q7pred = traj_pred_mean[:, [0]], traj_pred_mean[:, [1]], traj_pred_mean[:, [2]], traj_pred_mean[:, [3]], \
                                                                         traj_pred_mean[:, [4]], traj_pred_mean[:, [5]], traj_pred_mean[:, [6]]
                q1rightpred, q2rightpred, q3rightpred, q4rightpred, q5rightpred, q6rightpred, q7rightpred = right_traj_3_pred[:, [0]], right_traj_3_pred[:, [1]], right_traj_3_pred[:, [2]], \
                                                                                                            right_traj_3_pred[:, [3]], right_traj_3_pred[:, [4]], right_traj_3_pred[:, [5]], \
                                                                                                            right_traj_3_pred[:, [6]]
                q1leftpred, q2leftpred, q3leftpred, q4leftpred, q5leftpred, q6leftpred, q7leftpred = left_traj_3_pred[:,[0]], left_traj_3_pred[:, [1]], left_traj_3_pred[:,[2]], \
                                                                                                     left_traj_3_pred[:, [3]], left_traj_3_pred[:,[4]], left_traj_3_pred[:, [5]], \
                                                                                                     left_traj_3_pred[:,[6]]
                q1rightpred_1, q2rightpred_1, q3rightpred_1, q4rightpred_1, q5rightpred_1, q6rightpred_1, q7rightpred_1 = right_traj_1_pred[:,[0]], right_traj_1_pred[:,[1]], right_traj_1_pred[:,[2]],\
                                                                                                                          right_traj_1_pred[:,[3]], right_traj_1_pred[:,[4]], right_traj_1_pred[:,[5]], \
                                                                                                                          right_traj_1_pred[:,[6]]
                q1leftpred_1, q2leftpred_1, q3leftpred_1, q4leftpred_1, q5leftpred_1, q6leftpred_1, q7leftpred_1 = left_traj_1_pred[:, [0]], left_traj_1_pred[:,[1]],left_traj_1_pred[:, [2]], \
                                                                                                                   left_traj_1_pred[:,[3]], left_traj_1_pred[:, [4]], left_traj_1_pred[:,[5]], \
                                                                                                                   left_traj_1_pred[:, [6]]

            fig, axarr = plt.subplots(2, 4, figsize=(8, 3))
            axarr[0, 0].tick_params(axis='both', labelsize=5)
            axarr[0, 1].tick_params(axis='both', labelsize=5)
            axarr[0, 2].tick_params(axis='both', labelsize=5)
            axarr[0, 3].tick_params(axis='both', labelsize=5)
            axarr[1, 0].tick_params(axis='both', labelsize=5)
            axarr[1, 1].tick_params(axis='both', labelsize=5)
            axarr[1, 2].tick_params(axis='both', labelsize=5)
            axarr[1, 3].tick_params(axis='both', labelsize=5)
            fig.suptitle('Trajectories distributions configuration  ' + str(config), fontweight="bold")
            # Q1

            plt.sca(axarr[0, 0])
            x = np.linspace(0, 150, 150)
            plt.plot(q1true, 'c', label='q1 true', linewidth=0.5)
            plt.legend(loc=1, fontsize='x-small')
            plt.plot(q1righttrue, 'b', linewidth=0.5)
            plt.plot(q1lefttrue, 'b', linewidth=0.5)
            plt.plot(q1righttrue_1, 'b', linewidth=0.5)
            plt.plot(q1lefttrue_1, 'b', linewidth=0.5)
            axarr[0, 0].fill_between(x, q1righttrue.reshape(150, ), q1lefttrue.reshape(150, ), alpha=0.25,facecolor='blue')
            axarr[0, 0].fill_between(x, q1righttrue_1.reshape(150, ), q1lefttrue_1.reshape(150, ), alpha=0.25,facecolor='blue')

            if mean_traj_2.all() != 0:
                plt.plot(q1pred, 'r', label='q1 pred', linewidth=0.5)
                plt.plot(q1rightpred, 'm', linewidth=0.5)
                plt.plot(q1leftpred, 'm', linewidth=0.5)
                plt.plot(q1rightpred_1, 'm', linewidth=0.5)
                plt.plot(q1leftpred_1, 'm', linewidth=0.5)
                axarr[0, 0].fill_between(x, q1rightpred.reshape(150, ), q1leftpred.reshape(150, ), alpha=0.25,facecolor=(1, 0, 0, .4))
                axarr[0, 0].fill_between(x, q1rightpred_1.reshape(150, ), q1leftpred_1.reshape(150, ), alpha=0.25,facecolor=(1, 0, 0, .4))
            plt.legend(loc=1, fontsize='x-small')

            #Q2
            plt.sca(axarr[0, 1])
            x = np.linspace(0, 150, 150)
            plt.plot(q2true, 'c', label='q2 true', linewidth=0.5)
            plt.legend(loc=1, fontsize='x-small')
            plt.plot(q2righttrue, 'b', linewidth=0.5)
            plt.plot(q2lefttrue, 'b', linewidth=0.5)
            plt.plot(q2righttrue_1, 'b', linewidth=0.5)
            plt.plot(q2lefttrue_1, 'b', linewidth=0.5)
            axarr[0, 1].fill_between(x, q2righttrue.reshape(150, ), q2lefttrue.reshape(150, ), alpha=0.25,facecolor='blue')
            axarr[0, 1].fill_between(x, q2righttrue_1.reshape(150, ), q2lefttrue_1.reshape(150, ), alpha=0.25,facecolor='blue')

            if mean_traj_2.all() != 0:
                plt.plot(q2pred, 'r', label='q2 pred', linewidth=0.5)
                plt.plot(q2rightpred, 'm', linewidth=0.5)
                plt.plot(q2leftpred, 'm', linewidth=0.5)
                plt.plot(q2rightpred_1, 'm', linewidth=0.5)
                plt.plot(q2leftpred_1, 'm', linewidth=0.5)
                axarr[0, 1].fill_between(x, q2rightpred.reshape(150, ), q2leftpred.reshape(150, ), alpha=0.25,facecolor=(1, 0, 0, .4))
                axarr[0, 1].fill_between(x, q2rightpred_1.reshape(150, ), q2leftpred_1.reshape(150, ), alpha=0.25,facecolor=(1, 0, 0, .4))
            plt.legend(loc=1, fontsize='x-small')
            # Q3
            plt.sca(axarr[0, 2])
            x = np.linspace(0, 150, 150)
            plt.plot(q3true, 'c', label='q3 true', linewidth=0.5)
            plt.legend(loc=1, fontsize='x-small')
            plt.plot(q3righttrue, 'b', linewidth=0.5)
            plt.plot(q3lefttrue, 'b', linewidth=0.5)
            plt.plot(q3righttrue_1, 'b', linewidth=0.5)
            plt.plot(q3lefttrue_1, 'b', linewidth=0.5)
            axarr[0, 2].fill_between(x, q3righttrue.reshape(150, ), q3lefttrue.reshape(150, ), alpha=0.25,facecolor='blue')
            axarr[0, 2].fill_between(x, q3righttrue_1.reshape(150, ), q3lefttrue_1.reshape(150, ), alpha=0.25,facecolor='blue')

            if mean_traj_2.all() != 0:
                plt.plot(q3pred, 'r', label='q3 pred', linewidth=0.5)
                plt.plot(q3rightpred, 'm', linewidth=0.5)
                plt.plot(q3leftpred, 'm', linewidth=0.5)
                plt.plot(q3rightpred_1, 'm', linewidth=0.5)
                plt.plot(q3leftpred_1, 'm', linewidth=0.5)
                axarr[0, 2].fill_between(x, q3rightpred.reshape(150, ), q3leftpred.reshape(150, ), alpha=0.25,facecolor=(1, 0, 0, .4))
                axarr[0, 2].fill_between(x, q3rightpred_1.reshape(150, ), q3leftpred_1.reshape(150, ), alpha=0.25,facecolor=(1, 0, 0, .4))
            plt.legend(loc=1, fontsize='x-small')
            # Q4
            plt.sca(axarr[0, 3])
            x = np.linspace(0, 150, 150)
            plt.plot(q4true, 'c', label='q4 true', linewidth=0.5)
            plt.legend(loc=1, fontsize='x-small')
            plt.plot(q4righttrue, 'b', linewidth=0.5)
            plt.plot(q4lefttrue, 'b', linewidth=0.5)
            plt.plot(q4righttrue_1, 'b', linewidth=0.5)
            plt.plot(q4lefttrue_1, 'b', linewidth=0.5)
            axarr[0, 3].fill_between(x, q4righttrue.reshape(150, ), q4lefttrue.reshape(150, ), alpha=0.25,facecolor='blue')
            axarr[0, 3].fill_between(x, q4righttrue_1.reshape(150, ), q4lefttrue_1.reshape(150, ), alpha=0.25,facecolor='blue')

            if mean_traj_2.all() != 0:
                plt.plot(q4pred, 'r', label='q4 pred', linewidth=0.5)
                plt.plot(q4rightpred, 'm', linewidth=0.5)
                plt.plot(q4leftpred, 'm', linewidth=0.5)
                plt.plot(q4rightpred_1, 'm', linewidth=0.5)
                plt.plot(q4leftpred_1, 'm', linewidth=0.5)
                axarr[0, 3].fill_between(x, q4rightpred.reshape(150, ), q4leftpred.reshape(150, ), alpha=0.25,facecolor=(1, 0, 0, .4))
                axarr[0, 3].fill_between(x, q4rightpred_1.reshape(150, ), q4leftpred_1.reshape(150, ), alpha=0.25,facecolor=(1, 0, 0, .4))
            plt.legend(loc=1, fontsize='x-small')
            # Q5
            plt.sca(axarr[1, 0])
            x = np.linspace(0, 150, 150)
            plt.plot(q5true, 'c', label='q5 true', linewidth=0.5)
            plt.legend(loc=1, fontsize='x-small')
            plt.plot(q5righttrue, 'b', linewidth=0.5)
            plt.plot(q5lefttrue, 'b', linewidth=0.5)
            plt.plot(q5righttrue_1, 'b', linewidth=0.5)
            plt.plot(q5lefttrue_1, 'b', linewidth=0.5)
            axarr[1, 0].fill_between(x, q5righttrue.reshape(150, ), q5lefttrue.reshape(150, ), alpha=0.25,facecolor='blue')
            axarr[1, 0].fill_between(x, q5righttrue_1.reshape(150, ), q5lefttrue_1.reshape(150, ), alpha=0.25,facecolor='blue')

            if mean_traj_2.all() != 0:
                plt.plot(q5pred, 'r', label='q5 pred', linewidth=0.5)
                plt.plot(q5rightpred, 'm', linewidth=0.5)
                plt.plot(q5leftpred, 'm', linewidth=0.5)
                plt.plot(q5rightpred_1, 'm', linewidth=0.5)
                plt.plot(q5leftpred_1, 'm', linewidth=0.5)
                axarr[1, 0].fill_between(x, q5rightpred.reshape(150, ), q5leftpred.reshape(150, ), alpha=0.25,facecolor=(1, 0, 0, .4))
                axarr[1, 0].fill_between(x, q5rightpred_1.reshape(150, ), q5leftpred_1.reshape(150, ), alpha=0.25,facecolor=(1, 0, 0, .4))
            plt.legend(loc=1, fontsize='x-small')
            # Q6
            plt.sca(axarr[1, 1])
            x = np.linspace(0, 150, 150)
            plt.plot(q6true, 'c', label='q6 true', linewidth=0.5)
            plt.plot(q6righttrue, 'b', linewidth=0.5)
            plt.plot(q6lefttrue, 'b', linewidth=0.5)
            plt.plot(q6righttrue_1, 'b', linewidth=0.5)
            plt.plot(q6lefttrue_1, 'b', linewidth=0.5)
            axarr[1, 1].fill_between(x, q6righttrue.reshape(150, ), q6lefttrue.reshape(150, ), alpha=0.25,facecolor='blue')
            axarr[1, 1].fill_between(x, q6righttrue_1.reshape(150, ), q6lefttrue_1.reshape(150, ), alpha=0.25,facecolor='blue')

            if  mean_traj_2.all() != 0:
                plt.plot(q6pred, 'r', label='q6 pred', linewidth=0.5)
                plt.plot(q6rightpred, 'm', linewidth=0.5)
                plt.plot(q6leftpred, 'm', linewidth=0.5)
                plt.plot(q6rightpred_1, 'm', linewidth=0.5)
                plt.plot(q6leftpred_1, 'm', linewidth=0.5)
                axarr[1, 1].fill_between(x, q6rightpred.reshape(150, ), q6leftpred.reshape(150, ), alpha=0.25,facecolor=(1, 0, 0, .4))
                axarr[1, 1].fill_between(x, q6rightpred_1.reshape(150, ), q6leftpred_1.reshape(150, ), alpha=0.25,facecolor=(1, 0, 0, .4))
            plt.legend(loc=1, fontsize='x-small')
            # Q7
            plt.sca(axarr[1, 2])
            x = np.linspace(0, 150, 150)
            plt.plot(q7true, 'c', label='q7 true', linewidth=0.5)
            plt.plot(q7righttrue, 'b', linewidth=0.5)
            plt.plot(q7lefttrue, 'b', linewidth=0.5)
            plt.plot(q7righttrue_1, 'b', linewidth=0.5)
            plt.plot(q7lefttrue_1, 'b', linewidth=0.5)
            axarr[1, 2].fill_between(x, q7righttrue.reshape(150, ), q7lefttrue.reshape(150, ), alpha=0.25,facecolor='blue')
            axarr[1, 2].fill_between(x, q7righttrue_1.reshape(150, ), q7lefttrue_1.reshape(150, ), alpha=0.25,facecolor='blue')

            if  mean_traj_2.all() != 0:
                plt.plot(q7pred, 'r', label='q7 pred', linewidth=0.5)
                plt.plot(q7rightpred, 'm', linewidth=0.5)
                plt.plot(q7leftpred, 'm', linewidth=0.5)
                plt.plot(q7rightpred_1, 'm', linewidth=0.5)
                plt.plot(q7leftpred_1, 'm', linewidth=0.5)
                axarr[1, 2].fill_between(x, q7rightpred.reshape(150, ), q7leftpred.reshape(150, ), alpha=0.25,facecolor=(1, 0, 0, .4))
                axarr[1, 2].fill_between(x, q7rightpred_1.reshape(150, ), q7leftpred_1.reshape(150, ), alpha=0.25,facecolor=(1, 0, 0, .4))
                plt.legend(loc=1, fontsize='x-small')
            fig.set_dpi(200)
            if show==True:
                plt.show()  # Show the image.
            if save==True:
                # Create the path for saving plots.
                Path(save_path).mkdir(exist_ok=True, parents=True)
                plt.savefig(save_path+'joint_traj_distrib' + '.png')

def plot_mulivariate_weigths(save_file_path,mean_weights,covariance_weights,n_dof,n_func,show=False,save=False):
    sample_of_weights = np.random.multivariate_normal(mean_weights, covariance_weights, 100).T
    x = np.linspace(1, n_func * n_dof, num=n_func * n_dof)  # (8,0)
    fig,ax = plt.subplots(figsize=(17, 3))
    ax.grid()
    ax.plot(x, sample_of_weights, '.b')
    ax.plot(x, mean_weights, 'or')
    # Set tick font size
    for label in (ax.get_xticklabels()):
        label.set_fontsize(8)
    ax.set_xlabel('Weights', fontsize=8)
    plt.xticks(x)
    fig.set_dpi(100)
    if show==True:
        plt.show()  # Show the image.
    #Create the path for saving plots.
    if save==True:
        Path(save_file_path).mkdir(exist_ok=True, parents=True)
        plt.savefig(save_file_path + 'weights_distrib_multivariate' + '.png')

def plot_weights_distributions(save_file_path ,mean_weights_1,std_weights_1,n_dof,n_func,mean_weights_2=np.zeros(shape=(56,)),std_weights_2=np.zeros(shape=(56,)),show=False,save=False):
    x = np.linspace(1, n_func, num=n_func)  # (8,0)
    n_joints = n_dof
    columns = 3
    rows = int(np.ceil(n_joints / columns))
    fig, axarr = plt.subplots(rows, columns, figsize=(n_func, columns * 2))
    fig.tight_layout()
    fig.suptitle('Weights distributions ', fontweight="bold", fontsize=7)
    for i in range(n_dof):
        row = i // columns
        col = i % columns
        plt.sca(axarr[row, col])
        plt.bar(x, mean_weights_1[ 0 + i:i + n_func], yerr=std_weights_1[ 0 + i:i + n_func], align='center', alpha=0.5, ecolor='black', capsize=5)
        if mean_weights_2.all()!= 0 and std_weights_2.all()!= 0:
         plt.bar(x, mean_weights_2[0 + i:i + n_func], yerr=std_weights_2[0 + i:i + n_func],align='center', alpha=0.8, ecolor='red', color=(1, 0, 0, .4), capsize=5)
    if show == True:
        plt.show()  # Show the image.
    # Create the path for saving plots.
    if save == True:
        Path().mkdir(exist_ok=True, parents=True)
        plt.savefig(save_file_path + 'weights_distrib' + '.png')

def plot_traj_distribution_1_joint(save_path,config,xyz, mean_traj_1,traj_std_1,mean_traj_2=np.zeros(shape=(150,)),traj_std_2=np.zeros(shape=(150,)),show=False,save=False):
            #TRAJ 1
            traj_true_mean=mean_traj_1
            right_traj_1_true = mean_traj_1+traj_std_1
            left_traj_1_true = mean_traj_1-traj_std_1
            right_traj_3_true=mean_traj_1+3*traj_std_1
            left_traj_3_true = mean_traj_1-3*traj_std_1

            #TRAJ 2
            if  mean_traj_2.all() != 0:
                traj_pred_mean=mean_traj_2
                right_traj_1_pred = mean_traj_2+traj_std_2
                left_traj_1_pred = mean_traj_2-traj_std_2
                right_traj_3_pred =mean_traj_2+3*traj_std_2
                left_traj_3_pred = mean_traj_2-3*traj_std_2

            q1true= traj_true_mean
            q1righttrue = right_traj_3_true
            q1lefttrue = left_traj_3_true
            q1righttrue_1 = right_traj_1_true
            q1lefttrue_1= left_traj_1_true
            if  mean_traj_2.all() != 0:
                q1pred = traj_pred_mean
                q1rightpred = right_traj_3_pred
                q1leftpred = left_traj_3_pred
                q1rightpred_1= right_traj_1_pred
                q1leftpred_1 = left_traj_1_pred

            fig= plt.figure(figsize=(8, 3))
            fig.suptitle('Trajectories distributions configuration  ' + str(config), fontweight="bold")
            # Q1
            x = np.linspace(0, 150, 150)
            plt.plot(q1true, 'c', label='q7 true', linewidth=0.5)
            plt.legend(loc=1, fontsize='x-small')
            plt.plot(q1righttrue, 'b', linewidth=0.5)
            plt.plot(q1lefttrue, 'b', linewidth=0.5)
            plt.plot(q1righttrue_1, 'b', linewidth=0.5)
            plt.plot(q1lefttrue_1, 'b', linewidth=0.5)
            plt.fill_between(x, q1righttrue, q1lefttrue, alpha=0.25,facecolor='blue')
            plt.fill_between(x, q1righttrue_1, q1lefttrue_1, alpha=0.25,facecolor='blue')

            if mean_traj_2.all() != 0:
                plt.plot(q1pred, 'r', label='q7 pred', linewidth=0.5)
                plt.plot(q1rightpred, 'm', linewidth=0.5)
                plt.plot(q1leftpred, 'm', linewidth=0.5)
                plt.plot(q1rightpred_1, 'm', linewidth=0.5)
                plt.plot(q1leftpred_1, 'm', linewidth=0.5)
                #plt.fill_between(x, q1rightpred.reshape(150, ), q1leftpred.reshape(150, ), alpha=0.25,facecolor=(1, 0, 0, .4))
                #plt.fill_between(x, q1rightpred_1.reshape(150, ), q1leftpred_1.reshape(150, ), alpha=0.25,facecolor=(1, 0, 0, .4))
            plt.legend(loc=1, fontsize='x-small')

            fig.set_dpi(200)
            if show == True:
                plt.show()  # Show the image.
            if save == True:
                # Create the path for saving plots.
                Path(save_path+'/'+str(config)+'/').mkdir(exist_ok=True, parents=True)
                plt.savefig(save_path+'/'+str(config)+'/'+ str(xyz) +'.png')

def plot_weights_distributions_1_joint(save_file_path ,mean_weights_1,std_weights_1,n_func,mean_weights_2=np.zeros(shape=(8,)),std_weights_2=np.zeros(shape=(8,)),show=True,save=False):
    x = np.linspace(1, n_func, num=n_func)  # (8,0)
    fig = plt.figure()
    fig.tight_layout()
    fig.suptitle('Weights distributions configuration  ', fontweight="bold", fontsize=7)
    plt.bar(x, mean_weights_1, yerr=std_weights_1, align='center', alpha=0.5, ecolor='black', capsize=5)
    if mean_weights_2.all()!= 0 and std_weights_2.all()!= 0:
         plt.bar(x, mean_weights_2, yerr=std_weights_2,align='center', alpha=0.8, ecolor='red', color=(1, 0, 0, .4), capsize=5)
    fig.set_dpi(200)
    if show == True:
        plt.show()  # Show the image.
    # Create the path for saving plots.
    if save == True:
        Path().mkdir(exist_ok=True, parents=True)
        plt.savefig(save_file_path + 'weights_distrib' + '.png')