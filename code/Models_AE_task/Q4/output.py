import os
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
import tensorflow_probability as tfp
tfd = tfp.distributions
import tensorflow as tf
from sklearn.metrics import mean_squared_error

'''Return a dictionary of all variables in a module'''
def get_variables_from_module(module):
    """
    Source: https://stackoverflow.com/a/28150307
    """
    return {key: value for key, value in module.__dict__.items() if not (key.startswith('__') or key.startswith('_'))}

''' Log class '''
class Log:
    def __init__(self, log_dir: str):
        self.datetime_created = datetime.now()
        self.log_dir = log_dir
        # Log dir creation.
        Path(log_dir).mkdir(exist_ok=True, parents=True)
        # Main log file creation.
        self.log_file = os.path.join(log_dir, 'log.txt')
        with open(self.log_file, "w") as f:
            f.write(f"Log file created on {self.datetime_created.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    def log(self, text: str):
        with open(self.log_file, "a") as f:
            f.write(text)
            f.write("\n")

    def print(self, text: str):
        print(text)
        self.log(text)

    def get_file_path(self, filename: str):
        return os.path.join(self.log_dir, filename)

    def log_config(self, cfg_module):
        config_book = get_variables_from_module(cfg_module)
        with open(self.get_file_path('config.txt'), "w") as f:
            for key, value in config_book.items():
                f.write(f"{key:30}{str(value):100}\n")

''' Plot loss function '''
def plot_loss(plot_path, loss, name,val_loss=None):
    '''
    Plot of the history loss for this training cycle.
    :param plot_path: Save folder of the plot of the loss
    :param loss: Training loss
    :param val_loss: Validation loss
    '''
    fig = plt.figure()
    plt.plot(loss, 'r', label='train')
    if val_loss:
        plt.plot(val_loss, 'b', label='val')
    plt.grid(True, which='both')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(plot_path, name))
    plt.close(fig)
    # Same, but logarithmic y scale.
    fig = plt.figure()
    plt.plot(loss, 'r', label='train')
    if val_loss:
        plt.plot(val_loss, 'b', label='val')
    plt.yscale('log')
    plt.grid(True, which='both')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join(plot_path, 'log_'+name))
    plt.close(fig)

''' Metric function'''
def My_metric(mean_pred, mean_true, cov_pred,cov_true):

    mse_mean = mean_squared_error(mean_true, mean_pred)
    mse_cov = mean_squared_error(cov_true, cov_pred)
    return mse_cov + 0.5 *mse_mean


def My_metric_final_point(final_mean, final_mean_pred, final_right,final_right_pred):

    mse_mean = mean_squared_error(final_mean, final_mean_pred)
    mse_cov = mean_squared_error(final_right, final_right_pred)
    return mse_cov + mse_mean
