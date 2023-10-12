import os

import pomegranate.hmm as hmm
import pomegranate.distributions as distributions
import random
import numpy as np
from torch.masked import masked_tensor

from Models.model_wrapper import model_wrapper
import pickle
import inspect
import torch


class pome_wrapper(model_wrapper):
    def __init__(self, n_components, n_iter, distribution, dims, emission_prob=None, freeze_distributions=False):
        super().__init__(n_components, n_iter)
        self._model: hmm.DenseHMM = None
        self.emission_prob = emission_prob
        self.freeze_distributions = freeze_distributions
        self.distribution = distribution
        self.dims = dims
        # create model
        self.generate_initial_model_pytorch()

        data_dir = os.path.dirname(inspect.getfile(model_wrapper))
        self.cache_dir = os.path.join(data_dir, 'Cache')

    def generate_initial_model_pytorch(self):
        dists = self.create_distributions()
        tensor_type = torch.float32
        edges = torch.rand(self.n_components, self.n_components, dtype=tensor_type)
        starts = torch.rand(self.n_components, dtype=tensor_type)
        starts = starts / torch.sum(starts)
        ends = torch.rand(self.n_components, dtype=tensor_type)
        self._model = hmm.DenseHMM(dists, edges=edges, starts=starts, ends=ends, max_iter=self.n_iter)

    def fit(self, data, omission_idx=None):
        if omission_idx is not None:
            masked_data = []
            for sentence, ws in zip(data, omission_idx):
                sentence_with_zeros = np.full(max(ws) + 1, 0)
                sentence_with_zeros[ws] = sentence
                mask = np.full(max(ws) + 1, False)
                mask[ws] = True
                masked_data.append(torch.masked.MaskedTensor(torch.from_numpy(sentence_with_zeros), torch.from_numpy(mask)).reshape(-1, 1))
            data = masked_data
        else:
            data = [torch.from_numpy(arr).reshape(-1, 1) for arr in data]

        data = self.partition_sequences(data)
        self._model.fit(X=data)
        dists = self._model.distributions
        means = [dist.means for dist in dists]

    @property
    def transmat(self):
        try:
            return torch.exp(self._model.edges)
        except AttributeError as e:
            print(f"Model not initialized with fit, exception was raised: {e}")

    @property
    def startprob(self):
        try:
            return torch.exp(self._model.starts)
        except AttributeError as e:
            print(f"Model not initialized with fit, exception was raised: {e}")

    @property
    def emissionprob(self):
        return self.create_emissionprob()

    def create_emissionprob(self):
        emission_prob = []
        for prob in self._model.distributions:
            emission_prob.append(np.array(prob.probs))

        return np.vstack(emission_prob)

    def partition_sequences(self, data):
        lengths_dict = {}
        for tensor in data:
            if tensor.shape[0] not in lengths_dict:
                lengths_dict[tensor.shape[0]] = torch.unsqueeze(tensor, dim=0)
            else:
                lengths_dict[tensor.shape[0]] = torch.cat((lengths_dict[tensor.shape[0]], torch.unsqueeze(tensor, dim=0)))
        values = list(lengths_dict.values())
        values = sorted(values, key=lambda x: x.shape[1])
        return values

    def create_distributions(self):
        if self.distribution == 'Categorical':
            # return [distributions.Categorical(torch.from_numpy(np.atleast_2d(dist)), frozen=self.freeze_distributions)
            # for dist in self.emission_prob]
             return [distributions.Categorical(frozen=self.freeze_distributions)
                    for i in range(self.n_components)]
        elif self.distribution == 'Gaussian':
            means, covs = self.generate_initial_normal_params_pytorch()
            return [distributions.Normal(means=means[i], covs=covs[i], frozen=self.freeze_distributions) for i in
                    range(self.n_components)]
        else:
            raise NotImplementedError(f"No model implemented for {self.distribution} distribution")

    def generate_initial_normal_params_pytorch(self):
        tensor_type = torch.float32
        means = [torch.rand(self.dims, dtype=tensor_type) for i in range(self.n_components)]

        random_matrix = torch.randn(self.dims, self.dims)

        covariance_matrices = [torch.mm(random_matrix, random_matrix.t()) for i in range(self.n_components)]
        covariance_matrices = [covariance_matrix / torch.diag(covariance_matrix).sqrt().view(-1, 1) for
                               covariance_matrix in covariance_matrices]
        covariance_matrices = [covariance_matrix / torch.diag(covariance_matrix).sqrt().view(1, -1) for
                               covariance_matrix in covariance_matrices]

        return means, covariance_matrices


"""
    def add_states_with_dist(self):
        dist_dict = self.create_dist_dict()
        states = []
        # creating states from dist and dict
        for i, state_dist in enumerate(dist_dict):
            dist = DiscreteDistribution(state_dist)
            states.append(State(dist, name=str(i)))
        # adding states
        self.states = states
        self.full_states = self.states + [self._model.start, self._model.end]
        self._model.add_states(self.states)

    def add_start_and_end_transitions(self):
        for state in self.states:
            self._model.add_transition(self._model.start, state, random.uniform(0, 1))
        for state in self.states:
            self._model.add_transition(state, self._model.end, random.uniform(0, 1))
        pass

    def add_random_transitions(self):
        self.add_start_and_end_transitions()
        for state_a in self.states:
            for state_b in self.states:
                self._model.add_transition(state_a, state_b, random.uniform(0, 1))





    def create_dist_dict(self):
        # Should create a dict of dist from the self.emission_prob, as seen in the example from ChatGPT:
        #     'distributions': [DiscreteDistribution({'A': 0.6, 'C': 0.1, 'G': 0.2, 'T': 0.1})
        #     , DiscreteDistribution({'A': 0.2, 'C': 0.3, 'G': 0.2, 'T': 0.3})],
        # for the init_params?
        state_probs = []
        for state_prob in self.emission_prob:
            state_probs.append(dict(enumerate(state_prob)))
        return state_probs
"""
