import scipy
import numpy as np
import tensorflow as tf
import tensorflow_probability.python as tfp
import tensorflow_probability.python.distributions as tfd

'''Given the tensor with the elements of the fake triangular matrix, obtain L and D.
    L and D are the matrices of LDL decomposition '''
def LD_from_elements_pred(lower_tri_elements: tf.Tensor):

    # Build the triangular matrix from the elements.
    lower_tri = tfp.math.fill_triangular(lower_tri_elements)
    # Extract the diagonal values.
    d = tf.linalg.band_part(lower_tri, 0, 0)
    # Exploit the diagonal to compute an identity matrix with the correct size.
    identity = tf.linalg.diag(tf.linalg.diag_part(d) / tf.linalg.diag_part(d))
    # Compute the L matrix (lower uni-traingular).
    L = lower_tri - d + identity
    # Compute the D matrix (diagonal).
    # Note. Since d is diagonal exp(d) causes all 0 values to become 1. We address that by removing 1 from
    # all elements, then adding 1 back to the diagonal elements with the identity.
    D = tf.math.exp(d) - tf.ones_like(d) + identity
    return L, D

""" Given the tensor with the elements of the fake triangular matrix, obtain L and D.
L and D are the matrices of LDL decomposition. """
def LD_from_elements_true(lower_tri_elements: tf.Tensor):
    # Build the triangular matrix from the elements.
    lower_tri = tfp.math.fill_triangular(lower_tri_elements)
    # Extract the diagonal values.
    d = tf.linalg.band_part(lower_tri, 0, 0)
    # Exploit the diagonal to compute an identity matrix with the correct size.
    identity = tf.linalg.diag(tf.linalg.diag_part(d) / tf.linalg.diag_part(d))
    # Compute the L matrix (lower uni-traingular).
    L = lower_tri - d + identity
    # Compute the D matrix (diagonal).
    # Note. Since d is diagonal exp(d) causes all 0 values to become 1. We address that by removing 1 from
    # all elements, then adding 1 back to the diagonal elements with the identity.
    D = d
    return L, D

'''Re-compose the matrix from L and D decomposition '''
def LD_reconstruct(L: tf.Tensor, D: tf.Tensor):
    """ L is lower uni-triangular. D is diagonal."""
    cov = tf.matmul(L, tf.matmul(D, tf.transpose(L, (0, 2, 1))))
    return cov
