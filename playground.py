import matplotlib.pyplot as plt
import torch
from numpy import random
from pomegranate import distributions
import numpy as np
import numpy as np
from scipy.linalg import block_diag


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

def print_temporal_info_for_power_matrix(stochastic_matrix):
    for i in range(1, 30):
        stochastic_matrix_power = np.linalg.matrix_power(stochastic_matrix, i)
        print(f"Info in iter:{i} = {find_temporal_info_ratio(stochastic_matrix_power)}")


n_components = 10
print_temporal_info_for_power_matrix(generate_near_biased_matrix_imporved(10))
print_temporal_info_for_power_matrix(generate_near_biased_matrix_imporved(10))



"""# Assign transition probabilities

stochastic_matrix = np.empty((num_states, num_states))
for i in range(num_states):
    for j in range(num_states):
        # Example: Higher probability for transitions between nearby states
        stochastic_matrix[i, j] = np.exp(-abs(i - j)) / np.sum(np.exp(-np.abs(np.arange(num_states) - i)))

# Ensure rows sum to 1
stochastic_matrix /= stochastic_matrix.sum(axis=1, keepdims=True)
"""

"""sigma = [[[0.3,0],[0,0.3]], [[2,0],[0,2]]]
mues = [[3,1], [1,3]]
print(mues[0])
print(sigma[0])
dist = distributions.Normal(np.atleast_1d(mues[1]).astype(float), np.atleast_2d(sigma[1]).astype(float))






print(dist.sample(5))"""
"""
for i in range(10):
    a10 = np.random.rand(30,30)
    a10_sums = np.sum(a10, axis=0)
    a10 /= a10_sums.reshape(-1,1)

    b10 = np.random.rand(30,30)
    b10_sums = np.sum(b10, axis=0)
    b10 /= b10_sums.reshape(-1,1)

    a3 = np.random.rand(3,3)
    a3_sums = np.sum(a3, axis=0)
    a3 /= a3_sums.reshape(-1,1)

    b3 = np.random.rand(3,3)
    b3_sums = np.sum(b3, axis=0)
    b3 /= b3_sums.reshape(-1,1)

    print(f"for {a10.shape[0]} states: {np.sum(np.abs(a10-b10)) / a10.shape[0]}")

    print(f"for {a3.shape[0]} states: {np.sum(np.abs(a3-b3)) / a3.shape[0]}")
"""

"""
# creating the hmm samples from random
model = hmm.GaussianHMM(n_components=3 ,n_iter=20, covariance_type="full")
model.startprob_ = generate_random_normalized_vector(3)
model.transmat_ = generate_random_normalized_matrix((3,3))
model.means_ = np.expand_dims(np.random.uniform(1, 10, size=3), axis=1)
model.covars_ = np.tile(np.identity(1), (3, 1, 1))
data,Z = model.sample(100000)
data = np.array_split(data.astype(np.float32), data.shape[0]//10)
"""

"""# creating the hmm samples from random
model = hmm.GaussianHMM(n_components=3, covariance_type="full")
model.startprob_ = np.array([0, 0.2, 0.8])
model.transmat_ = np.array([[0, 0.5, 0.5], [0.5, 0.1, 0.4], [0.1, 0.5, 0.4]])
model.means_ = np.array([[10], [20], [30]])
model.covars_ = np.tile(0.01*np.identity(1), (3, 1, 1))
X, Z = model.sample(100000)
# data = np.array_split(X.astype(np.float32), X.shape[0]//10)


# creating the hmm model

model_learn = hmm.GaussianHMM(n_components=3, n_iter=20)
# data_squeezed = [np.squeeze(sentence) for sentence in data]
# data_hmmlearn_formatted = np.hstack(data_squeezed)
# data_hmmlearn_formatted = data_hmmlearn_formatted.reshape(-1, 1)
# sentences_length = [int(sentence.shape[0]) for sentence in data]
sentences_length = [10] * 10000
print(X)
print(sentences_length)
model_learn.fit(X, sentences_length)
print(model.transmat_)
print(model_learn.transmat_)
print(evaluations.compare_mat_l1_norm(model_learn.transmat_, model.transmat_))
print(model_learn.means_)
"""
"""
# %% create markov chain

SENTENCE_NUM = 100000
typical_sentence_len = 20
typical_sentence_len_var = 1
initial_dist = [0, 0.2, 0.8]
transition_matrix = [[0, 0.5, 0.5], [0.5, 0.1, 0.4], [0.1, 0.5, 0.4]]
sentences = []
sentences_len = np.random.normal(typical_sentence_len, typical_sentence_len_var, SENTENCE_NUM).astype(int)
alternate_length = typical_sentence_len * np.ones(sentences_len.shape).astype(int)
sentences_len = np.where(sentences_len > 0, sentences_len, alternate_length)

for sentence_len in sentences_len:

    sentence = []
    initial_state = np.random.choice([1, 2, 3], p=initial_dist)
    current_state = initial_state

    for _ in range(sentence_len):
        sentence.append(current_state)
        current_state = np.random.choice([1, 2, 3], p=transition_matrix[current_state - 1])
    sentences.append(sentence)

# %% create observations

A_mu = 10
B_mu = 20
C_mu = 30
mu = [A_mu, B_mu, C_mu]

A_sigma = 0.01
B_sigma = 0.01
C_sigma = 0.01
sigma = [A_sigma, B_sigma, C_sigma]

total_observations = []

for sentence in sentences:

    observations = []

    for word in sentence:
        observation = np.random.normal(mu[word - 1], sigma[word - 1])
        observations.append(observation)
    total_observations.append(observations)

# %% split train and test
n_states = 3

train = np.array([])
test = np.array([])
train_lengths = np.array([])
test_lengths = np.array([])

percent = 100

for observations in total_observations[:(percent * len(total_observations)) // 100]:
    train = np.append(train, observations)
    train_lengths = np.append(train_lengths, len(observations))

for observations in total_observations[1 + (percent * len(total_observations)) // 100:]:
    test = np.append(test, observations)
    test_lengths = np.append(test_lengths, len(observations))

train = train.astype(float)
test = test.astype(float)

# %% analyze HMM


model = hmm.GaussianHMM(n_components=n_states, n_iter=20)
model.fit(train.reshape(-1, 1), train_lengths.astype(int))
score = model.score(train.reshape(-1, 1), train_lengths.astype(int))
print(transition_matrix)
print(model.transmat_)
print(model.means_)
print(evaluations.compare_mat_l1_norm(model.transmat_, transition_matrix))

"""
"""
# %%


test_sentence1 = np.array([30, 10, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20])
test_sentence_states1 = test_sentence1 // 10
test_sentence2 = np.array([20, 30, 10, ])
test_sentence_states2 = test_sentence2 // 10
test_log_likelihood1, _ = model.score_samples(np.array(test_sentence1).reshape(-1, 1))
print(test_log_likelihood1)
print(MM_log_likelyhood(test_sentence_states1, transition_matrix, initial_dist))

test_log_likelihood2, _ = model.score_samples(np.array(test_sentence2).reshape(-1, 1))
print(test_log_likelihood2)
print(MM_log_likelyhood(test_sentence_states2, transition_matrix, initial_dist))

test_sentence = np.array([])
prev_log_likelyhood = 0
for i in range(15):
    test_sentence = np.append(test_sentence, 20)
    test_log_likelihood, _ = model.score_samples(np.array(test_sentence).reshape(-1, 1))
    print(test_log_likelihood - prev_log_likelyhood)
    prev_log_likelyhood = test_log_likelihood

test_sentence = np.array([])
prev_log_likelyhood = 0
for i in range(15):
    test_sentence = np.append(test_sentence, 30)
    test_log_likelihood, posterior = model.score_samples(np.array(test_sentence).reshape(-1, 1))
    print(test_log_likelihood - prev_log_likelyhood)
    prev_log_likelyhood = test_log_likelihood

result = -0.5 * (np.log(2 * np.pi)
                 + np.log(model.covars_[:, 0, 0]).sum(axis=-1)
                 + ((np.array(test_sentence).reshape(-1, 1)[:, None, :] - model.means_) ** 2 / model.covars_[:, 0,
                                                                                               0]).sum(axis=-1))
print(result)

result = -0.5 * (np.log(2 * np.pi)
                 + np.log(C_sigma ** 2)
                 + ((np.array(test_sentence).reshape(-1, 1)[:, None, :] - model.means_) ** 2 / model.covars_[:, 0,
                                                                                               0]).sum(axis=-1))
print(result)




# %% likelyhood function

def MM_log_likelyhood(X, T, start):
    log_likelyhood = np.log(start[X[0] - 1])

    for i in range(len(X) - 1):
        log_likelyhood += np.log(T[X[i] - 1][X[i + 1] - 1])

    return log_likelyhood


# %%
for observations in total_observations[1 + (percent * len(total_observations)) // 100:]:
    print()
"""
