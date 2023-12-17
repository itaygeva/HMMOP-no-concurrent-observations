import numpy as np


def generate_random_normalized_vector(length):
    vec = np.random.rand(length)
    vec = vec / np.sum(vec)
    return vec


def generate_random_normalized_matrix(shape):
    mat = np.random.random(shape)
    mat = mat / np.sum(mat, axis=1, keepdims=True)
    return mat
