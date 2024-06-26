import numpy as np
import torch
from itertools import permutations


def find_optimal_permutation_list(list, ref_list):
    return [find_optimal_permutation(list[i], ref_list[i]) for i in range(len(ref_list))]


def find_optimal_permutation(vec, ref_vec):
    min_norm = float('inf')
    optimal_permutation = None
    vec = np.squeeze(np.array(vec))
    ref_vec = np.squeeze(np.array(ref_vec))
    for perm in permutations(range(len(vec))):
        permuted_vec = np.array(vec)[list(perm)]
        current_norm = np.linalg.norm(permuted_vec - ref_vec)

        if current_norm < min_norm:
            min_norm = current_norm
            optimal_permutation = perm

    return optimal_permutation


def reorient_matrix(matrix, perm):
    matrix = matrix[perm, :]
    matrix = matrix[:, perm]
    return matrix


def reorient_matrix_list(matrix_list, perm_list):
    return [reorient_matrix(matrix_list[i], perm_list[i]) for i in range(len(matrix_list))]


def find_mat_diff(matrix1, matrix2):
    """
    finds the l1 distance between the matrices
    :param matrix1: torch tensor
    :param matrix2: a torch tensor to compare
    :return: The l1 distance between the matrices
    """
    return np.sum(np.abs(matrix1 - matrix2))


def compare_mat_l1_norm_for_list(list_of_matrix1, list_of_matrix2):
    """
    finds the l1 distance between the all matrix-elements of the lists
    :param list_of_matrix1: A list of matrices
    :param list_of_matrix2: Another list of matrices
    :return: A list of l1 distances
    """
    return [compare_mat_l1_norm(list_of_matrix1[i], list_of_matrix2[i]) for i in range(len(list_of_matrix1))]


def compare_mat_l1(matrix1, matrix2):
    """
    prints the l1 distance between the matrices
    :param matrix1: torch tensor
    :param matrix2: a torch tensor to compare
    """
    return find_mat_diff(matrix1, matrix2)


def compare_mat_l1_norm(matrix1, matrix2):
    """
    prints the normalized l1 distance between the matrices
    :param matrix1: torch tensor
    :param matrix2: a torch tensor to compare
    """
    return np.sum(np.abs(matrix1 - matrix2)) / matrix1.shape[0]


def find_temporal_info_ratio(matrix):
    """
    return the ratio of the sum of eigenvalues of value different to 1 over the sum of all eigenvalues.
    The larger this ratio is, the more temporal information is in the matrix.
    :param matrix: a matrix
    :return: the temporal ratio
    """
    eigenvalues = np.linalg.eigvals(matrix)
    temporal_eigenvalues_sum = np.sum(np.where(eigenvalues > 0.99, 0, np.abs(eigenvalues)))
    return temporal_eigenvalues_sum / np.sum(np.abs(eigenvalues))


