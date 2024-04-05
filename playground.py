import matplotlib.pyplot as plt
import torch
from numpy import random
from pomegranate import distributions
import numpy as np
import numpy as np
from scipy.linalg import block_diag
from Experiments.evalUtils import *


def find_temporal_info_ratio(matrix):
    """
    return the ratio of the sum of eigenvalues of value 1/ the sum of all eigenvalues.
    The larger this ratio is, the less temporal information is in the matrix.
    :param matrix: a matrix
    :return: the temporal ratio
    """
    eigenvalues = np.linalg.eigvals(matrix)
    temporal_eigenvalues_sum = np.sum(np.where(eigenvalues > 0.99, 0, np.abs(eigenvalues)))
    return temporal_eigenvalues_sum / np.sum(np.abs(eigenvalues))


def find_max_eig_ratio(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    eigenvalues = np.sort(np.abs(eigenvalues))
    return eigenvalues[-2]


def generate_stochastic_matrix_with_seed(n_states, seed):
    np.random.seed(seed)
    stochastic_matrix = np.random.rand(n_states, n_states)
    for line in stochastic_matrix:
        line /= np.sum(line)
    return stochastic_matrix


def generate_stochastic_matrix(n_states):
    stochastic_matrix = np.random.rand(n_states, n_states)
    for line in stochastic_matrix:
        line /= np.sum(line)
    return stochastic_matrix


def generate_block_diag_matrix(n_states, size_block):
    blocks = []
    for i in range(n_states // size_block):
        blocks.append(generate_stochastic_matrix(size_block))

    # Create a block diagonal matrix
    return block_diag(*blocks)


def generate_near_biased_matrix(n_states):
    stochastic_matrix = np.empty((n_states, n_states))
    for i in range(n_states):
        for j in range(n_states):
            # Example: Higher probability for transitions between nearby states
            stochastic_matrix[i, j] = np.exp(-abs(i - j)) / np.sum(np.exp(-np.abs(np.arange(n_states) - i)))

    # Ensure rows sum to 1
    stochastic_matrix /= stochastic_matrix.sum(axis=1, keepdims=True)
    return stochastic_matrix


def normalize_stochastic_matrix(stochastic_matrix):
    for row in stochastic_matrix:
        row /= np.sum(row)
    return stochastic_matrix


def generate_near_biased_matrix_imporved(n_states):
    rows = np.arange(n_states)
    # Create a 2D array with repeated rows to represent the column indices
    columns = np.tile(rows, (n_states, 1))
    # Calculate the absolute differences element-wise
    distance_matrix = -1 * np.abs(rows - columns.T)
    stochastic_matrix = np.exp(distance_matrix)

    return normalize_stochastic_matrix(stochastic_matrix)


def generate_n_edges_matrix(n_components, n_edges):
    stochastic_matrix = np.zeros((n_components, n_components))
    for row in stochastic_matrix:
        edges_idx = np.random.choice(n_components, size=(n_edges), replace=False)
        edges_values = np.random.random(n_edges)
        row[edges_idx] = edges_values

    return normalize_stochastic_matrix(stochastic_matrix)


def print_temporal_info_for_power_matrix(stochastic_matrix):
    for i in range(1, 30):
        stochastic_matrix_power = np.linalg.matrix_power(stochastic_matrix, i)
        print(f"Info in iter:{i} = {find_temporal_info_ratio(stochastic_matrix_power)}")


def print_max_eig_for_power_matrix(stochastic_matrix):
    for i in range(1, 30):
        stochastic_matrix_power = np.linalg.matrix_power(stochastic_matrix, i)
        print(f"Info in iter:{i} = {find_max_eig_ratio(stochastic_matrix_power)}")


n_components = 10


def averaged_L1_of_NN(n_states):
    results = []
    for i in range(10):
        stochastic_matrix = generate_stochastic_matrix(n_states)
        matrix = generate_near_biased_matrix_imporved(n_states)
        results.append(compare_mat_l1_norm(matrix, stochastic_matrix))
    return np.average(results)


def averaged_L1_of_N_Edges(n_states, n_edges):
    results = []
    for i in range(10):
        stochastic_matrix = generate_stochastic_matrix(n_states)
        matrix = generate_n_edges_matrix(n_states, n_edges)
        results.append(compare_mat_l1_norm(matrix, stochastic_matrix))
    return np.average(results)


def averaged_L1_of_Random(n_states):
    results = []
    for i in range(10):
        stochastic_matrix = generate_stochastic_matrix(n_states)
        matrix = generate_stochastic_matrix(n_states)
        results.append(compare_mat_l1_norm(matrix, stochastic_matrix))
    return np.average(results)


def compare_with_higher_powers(matrix, n_powers):
    for i in range(1, n_powers + 1):
        print(f"L1 power {i} = {compare_mat_l1_norm(matrix, np.linalg.matrix_power(matrix, i)):.2f}")

def averaged_L1_of_NN_for_powers(n_states, powers):

    results = np.empty((10, len(powers)))
    for i in range(10):
        stochastic_matrix = generate_stochastic_matrix(n_states)
        matrix = generate_near_biased_matrix_imporved(n_states)
        for k, power in enumerate(powers):
            results[i, k] = compare_mat_l1_norm(np.linalg.matrix_power(matrix, power), stochastic_matrix)
    results = np.mean(results, axis=0)
    for k, power in enumerate(powers):
        print(f"L1 of NN for power {power} : {results[k]:.2f}")
def averaged_L1_of_N_Edges_for_powers(n_states, powers):

    results = np.empty((10, len(powers)))
    for i in range(10):
        stochastic_matrix = generate_stochastic_matrix(n_states)
        matrix = generate_n_edges_matrix(n_states, 3)
        for k, power in enumerate(powers):
            results[i, k] = compare_mat_l1_norm(np.linalg.matrix_power(matrix, power), stochastic_matrix)
    results = np.mean(results, axis=0)
    for k, power in enumerate(powers):
        print(f"L1 of N_Edges for power {power} : {results[k]:.2f}")
def averaged_L1_of_Random_for_powers(n_states, powers):

    results = np.empty((10, len(powers)))
    for i in range(10):
        stochastic_matrix = generate_stochastic_matrix(n_states)
        matrix = generate_stochastic_matrix(n_states)
        for k, power in enumerate(powers):
            results[i, k] = compare_mat_l1_norm(np.linalg.matrix_power(matrix, power), stochastic_matrix)
    results = np.mean(results, axis=0)
    for k, power in enumerate(powers):
        print(f"L1 of Random for power {power} : {results[k]:.2f}")
arr = np.atleast_2d(np.ones(3, dtype=float)*1)
#print(arr.dtype)

np.set_printoptions(precision=2)
print(np.linalg.matrix_power(generate_near_biased_matrix(10),100))
print(np.linalg.matrix_power(generate_n_edges_matrix(10,3),100))
print(np.linalg.matrix_power(generate_stochastic_matrix(10),100))

