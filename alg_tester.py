import math
from Experiments import evalUtils
import numpy as np
from playground import *
from scipy.linalg import expm
from eq_solve_tester import *


def generate_transmat_powers_with_err(t, powers, err):
    t_powers = []
    for power in powers:
        t_powers.append(np.linalg.matrix_power(t, power) + (err / 2) * generate_stochastic_matrix(t1.shape[0]) -
                        (err / 2) * generate_stochastic_matrix(t1.shape[0]))
    return powers, t_powers


def generate_transmat_powers(t, powers):
    t_powers = []
    for power in powers:
        t_powers.append(np.linalg.matrix_power(t, power))
    return powers, t_powers


def generate_et(powers, matrices):
    et = np.zeros_like(matrices[0])
    for power, matrix in zip(powers, matrices):
        et += matrix * (1 / math.factorial(power))
    return et

err = 0
t1 = generate_stochastic_matrix_with_seed(3, 1)
et = expm(t1)
et_restored = generate_et(*generate_transmat_powers_with_err(t1, np.arange(2, 11), err))
et_restored_bad = generate_et(*generate_transmat_powers_with_err(t1, np.arange(2, 4), err))

print(et - t1 - np.linalg.matrix_power(t1, 0))
print(et_restored)
print(et_restored_bad)

eigval, eigvec = np.linalg.eig(et - t1 - np.linalg.matrix_power(t1, 0))
eigval_restored, eigvec_restored = np.linalg.eig(et_restored)
eigval_restored_bad, eigvec_restored_bad = np.linalg.eig(et_restored_bad)
eigval_og, eigvec_og = np.linalg.eig(t1)

print(eigval_og)
print(eigval)
print(eigval_restored)
print(eigval_restored_bad)

restored_og_eigvalues = []
for i, eigvalue in enumerate(eigval_restored):
    _, min_value, (x, y) = find_solution(eigvalue, [0, 1], 1, 200)
    print(f"{x} + {y}j with value {min_value} vs real eigvalue of {eigval_og[i]}")
    generate_min_image(eigvalue, [0, 1], 1, 200)
    restored_og_eigvalues.append(x + 1j * y)
print(restored_og_eigvalues)

lambda_mat = np.diag(restored_og_eigvalues)
reconstructed_t1 = np.abs(eigvec_restored @ lambda_mat @ np.linalg.inv(eigvec_restored))
print(reconstructed_t1)
print(t1)
print(evalUtils.compare_mat_l1_norm(t1, reconstructed_t1))


print(np.sum(reconstructed_t1, axis=1))