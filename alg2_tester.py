import math
from Experiments import evalUtils
import numpy as np
from playground import *
from scipy.linalg import expm
from eq_solve_tester import *
import cmath


def generate_transmat_powers_with_err(t, powers, err):
    t_powers = []
    for power in powers:
        t_powers.append(np.linalg.matrix_power(t, power) + (err / 2) * generate_stochastic_matrix(t1.shape[0]) -
                        (err / 2) * generate_stochastic_matrix(t1.shape[0]))
    return powers, t_powers


def find_roots(z, n):
    """
    Find the nth roots of a complex number.

    Parameters:
        z (complex): The complex number.
        n (int): The root order.

    Returns:
        list: A list containing the nth roots of the complex number.
    """
    roots = [z * cmath.exp(2j * k * cmath.pi / n) ** (1 / n) for k in range(n)]
    return roots


def find_eigenvalue(powers, eigenvalue_powers, n_iter=20, step=0.1):
    # We can find the relevant eigenvalue first and then do LLS, or we can use LLS to find the best eigenvalue
    possible_eigs = np.expand_dims(find_roots(eigenvalue_powers[0], powers[0]), axis=1)
    possible_eigs_powers = np.power(possible_eigs, powers)
    loss = possible_eigs_powers - eigenvalue_powers
    for i in range(n_iter):
        gradient = np.sum(np.sign(loss) * np.power(possible_eigs, powers - 1) * (-powers), axis=1)
        possible_eigs = np.expand_dims(possible_eigs.squeeze() - step * gradient, axis=1)
        possible_eigs_powers = np.power(possible_eigs, powers)
        loss = possible_eigs_powers - eigenvalue_powers
    risk = np.sum(np.abs(loss), axis=1)  # this is l1 and not normalized or weighted
    eig_idx = np.argmin(risk)
    return possible_eigs.squeeze()[eig_idx]


def get_power_eigenvalues(t_powers):
    eigvalue_powers = []
    for t_power in t_powers:
        eigvalue_powers.append(np.linalg.eigvals(t_powers))
    return np.array(eigvalue_powers).T


def find_eigenvalues(powers, eigvalue_powers):
    eigenvalues = []
    for single_eigenvalue_powers in eigvalue_powers:
        eigenvalues.append(find_eigenvalue(powers, single_eigenvalue_powers))
    return eigenvalues


def generate_transmat_powers(t, powers):
    t_powers = []
    for power in powers:
        t_powers.append(np.linalg.matrix_power(t, power))
    return powers, t_powers


err = 0
t1 = generate_stochastic_matrix_with_seed(3, 42)
eigs = np.linalg.eigvals(t1)

t_powers = generate_transmat_powers(t1, np.arange(2, 11))
restored_eigs = find_eigenvalues(np.arange(2, 11), t_powers)

print(eigs)
print(restored_eigs)
