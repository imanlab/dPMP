import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_probability as tfp
tfd = tfp.distributions
from LDL_decomposition import LD_reconstruct,LD_from_elements_pred,LD_from_elements_true
#np.set_printoptions(threshold=np.inf)
#tf.config.run_functions_eagerly(True)

def RMSE(promp):
    all_phi = tf.cast(promp.all_phi(), 'float64')
    def RMSE_loss(ytrue, ypred):

        mean_true = ytrue[..., 0:-36]
        lower_tri_elements_true = ytrue[..., -36:]
        mean_pred = ypred[..., 0:-36]
        lower_tri_elements_pred= ypred[..., -36:]

        mean_traj_true=tf.transpose(tf.matmul(all_phi, tf.transpose(mean_true)))
        mean_traj_pred=tf.transpose(tf.matmul(all_phi, tf.transpose(mean_pred)))

        L_true, D_true= LD_from_elements_true(lower_tri_elements_true)
        L_pred, D_pred= LD_from_elements_pred(lower_tri_elements_pred)

        cov_true = LD_reconstruct(L_true, D_true)
        cov_pred = LD_reconstruct(L_pred, D_pred)

        cov_traj_true =tf.linalg.matmul(cov_true, tf.transpose(all_phi))
        cov_traj_true=tf.linalg.matmul(tf.transpose(cov_traj_true, perm=[0, 2, 1]), tf.transpose(all_phi))
        cov_traj_pred =tf.linalg.matmul(cov_pred, tf.transpose(all_phi))
        cov_traj_pred=tf.linalg.matmul(tf.transpose(cov_traj_pred, perm=[0, 2, 1]), tf.transpose(all_phi))

        loss_mean = (K.mean(K.square(mean_traj_true - mean_traj_pred)))
        loss_cov = (K.mean(K.square(cov_traj_true - cov_traj_pred)))

        return loss_mean+1.5*loss_cov

    return RMSE_loss