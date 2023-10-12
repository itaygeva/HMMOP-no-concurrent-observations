import numpy as np
import torch

def find_mat_diff(matrix1, matrix2):
    matrix1 = matrix1.numpy()
    matrix2 = matrix2.numpy()
    return np.sum(np.abs(matrix1 - matrix2))

def compare_mat_l1(matrix1, matrix2):
    print(find_mat_diff(matrix1, matrix2))


def compare_mat_l1_norm(matrix1, matrix2):
    print(find_mat_diff(matrix1, matrix2)/matrix1.shape[0])


def find_temporal_info_ratio(matrix):
    """
    return the ratio of the sum of eigenvalues of value 1/ the sum of all eigenvalues.
    The larger this ratio is, the less temporal information is in the matrix.
    :param matrix: a matrix
    :return: the temporal ratio
    """
    eigenvalues = np.linalg.eigvals(matrix)
    number_of_ones = np.sum(np.count_nonzero(eigenvalues == 1))
    return number_of_ones/np.sum(np.abs(eigenvalues))
