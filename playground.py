import numpy as np
import pomegranate.hmm as hmm
import pomegranate.distributions as distributions
from Data.Readers.brown_corpus_reader import BCReader
import Omitters.utils as omitter
from Models.hmmlearn_wrapper import hmmlearn_wrapper
from Evaluations import utils as evaluations
import torch
from torch.masked import MaskedTensor
from numpy import random
import itertools
from Data.Readers.stocks_reader import StocksReader

"""def generate_initial_normal_params_pytorch(dims, n_components):
    tensor_type = torch.float32
    means = [torch.rand(dims, dtype=tensor_type) for i in range(n_components)]

    covariance_matrices = [torch.randn(dims, dims) for i in range(n_components)]

    covariance_matrices = [torch.mm(covariance_matrix, covariance_matrix.t()) for
                           covariance_matrix in covariance_matrices]
    covariance_matrices = [covariance_matrix / torch.diag(covariance_matrix).sqrt().view(-1, 1) for
                           covariance_matrix in covariance_matrices]
    covariance_matrices = [covariance_matrix / torch.diag(covariance_matrix).sqrt().view(1, -1) for
                           covariance_matrix in covariance_matrices]

    return means, covariance_matrices


def generate_initial_model_pytorch(distributions, n_components, n_iter):
    tensor_type = torch.float32
    edges = torch.rand(n_components, n_components, dtype=tensor_type)
    ends = torch.rand(n_components, dtype=tensor_type)

    for s in range(n_components):
        temp = np.random.choice(range(1000), n_components + 1, replace=False)
        edges[s, :] = torch.from_numpy(temp[:-1] / np.sum(temp))
        ends[s] = temp[-1] / np.sum(temp)

    starts = torch.rand(n_components, dtype=tensor_type)
    starts = starts / torch.sum(starts)

    return hmm.DenseHMM(distributions, edges=edges, starts=starts, ends=ends, max_iter=n_iter)


def partition_sequences(data):
    lengths_dict = {}
    for tensor in data:
        if tensor.shape[0] not in lengths_dict:
            lengths_dict[tensor.shape[0]] = torch.unsqueeze(tensor, dim=0)
        else:
            lengths_dict[tensor.shape[0]] = torch.cat(
                (lengths_dict[tensor.shape[0]], torch.unsqueeze(tensor, dim=0)))
    values = list(lengths_dict.values())
    values = sorted(values, key=lambda x: x.shape[1])
    return values


n_iter = 100
dims = 1
n_states = 3
freeze_distributions = False
sample_num = 1000000

means, covs = generate_initial_normal_params_pytorch(dims, n_states)
dists = [distributions.Normal(means=means[i], covs=covs[i], frozen=freeze_distributions) for i in
         range(n_states)]

#samples = [dist.sample(sample_num) for dist in dists]

tester_means, tester_covs = generate_initial_normal_params_pytorch(dims, n_states)
tester_dists = [distributions.Normal(means=tester_means[i], covs=tester_covs[i], frozen=freeze_distributions) for i in
                range(n_states)]"""
"""for i, dist in enumerate(tester_dists):
    dist.fit(samples[i])

print("means:")
print([dist.means for dist in dists])
print([dist.means for dist in tester_dists])
print("covs")
print([dist.covs for dist in dists])
print([dist.covs for dist in tester_dists])
"""
"""model = generate_initial_model_pytorch(dists, n_states, n_iter)
samples = model.sample(sample_num)
samples = partition_sequences(samples)
samples = [samples[i] for i in range(1, len(samples))]



tester_model = generate_initial_model_pytorch(tester_dists, n_states, n_iter)
tester_model.fit(samples)

transmat = torch.exp(model.edges)
tester_transmat = torch.exp(tester_model.edges)

means = [dist.means for dist in model.distributions]
tester_means = [dist.means for dist in tester_model.distributions]

covs = [dist.covs for dist in model.distributions]
tester_covs = [dist.covs for dist in tester_model.distributions]

print("transition matrix:")
print(transmat)
print(tester_transmat)

print("means:")
print(means)
print(tester_means)

print("covs:")
print(covs)
print(tester_covs)"""

x1 = np.linspace(0, 1, 10)
x2 = [StocksReader(), StocksReader(), StocksReader()]
x3 = 1
grid = np.squeeze(np.meshgrid(x1, x2, x3))
print(grid)


