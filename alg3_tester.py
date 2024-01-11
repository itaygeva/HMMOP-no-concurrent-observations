import math
from Experiments import evalUtils
import numpy as np
from playground import *
from scipy.linalg import expm
from eq_solve_tester import *
import cmath
from egcd import egcd


def get_egcd_coeff(powers):
    gcd = powers[0]
    coeff = np.ones(len(powers))
    for i in range(1, len(powers)):
        gcd, x, y = egcd(gcd, powers[i])
        coeff_components = np.ones_like(coeff)
        coeff_components[0:i] = x
        coeff_components[i] = y
        coeff *= coeff_components
    return coeff


def generate_transmat_powers_with_err(t, powers, err):
    t_powers = []
    for power in powers:
        t_powers.append(np.linalg.matrix_power(t, power) + (err / 2) * generate_stochastic_matrix(t.shape[0]) -
                        (err / 2) * generate_stochastic_matrix(t.shape[0]))
    return powers, t_powers


def generate_transmat_powers(t, powers):
    t_powers = []
    for power in powers:
        t_powers.append(np.linalg.matrix_power(t, power))
    return powers, t_powers


def restore_t(powers, t_powers):
    coefficients = get_egcd_coeff(powers)
    t = np.identity(t_powers[0].shape[0])
    for coeff, t_power in zip(coefficients, t_powers):
        temp = np.linalg.matrix_power(t_power, np.abs(coeff).astype(int))
        temp = np.linalg.inv(temp) if coeff < 0 else temp
        t = t @ temp
    return t


err = 0.1
t1 = generate_stochastic_matrix_with_seed(3, 42)
powers = np.array([2, 3])

_, t_powers = generate_transmat_powers_with_err(t1, powers, err)
t1_restored = restore_t(powers, t_powers)
print(t1)
print(t1_restored)
