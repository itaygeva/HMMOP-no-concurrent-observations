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
    def __init__(self, n_components, n_iter, emission_prob, freeze_distributions=False):
        super().__init__(n_components, n_iter)
        self._model: hmm.DenseHMM = None
        self.emission_prob = emission_prob
        self.freeze_distributions = freeze_distributions

        # create model
        self.generate_initial_model_pytorch()

        data_dir = os.path.dirname(inspect.getfile(model_wrapper))
        self.cache_dir = os.path.join(data_dir, 'Cache')

    def generate_initial_model_pytorch(self):
        dists = [distributions.Categorical(torch.from_numpy(np.atleast_2d(dist)), frozen=self.freeze_distributions)
                 for dist in self.emission_prob]
        tensor_type = torch.float32
        edges = torch.rand(self.n_components, self.n_components, dtype=tensor_type)
        starts = torch.rand(self.n_components, dtype=tensor_type)
        starts = starts / torch.sum(starts)
        ends = torch.rand(self.n_components, dtype=tensor_type)
        self._model = hmm.DenseHMM(dists, edges=edges, starts=starts, ends=ends, max_iter=self.n_iter)

    def generate_initial_model(self):
        dists = [distributions.Categorical(np.atleast_2d(dist), frozen=self.freeze_distributions)
                 for dist in self.emission_prob]
        edges = np.random.rand(self.n_components, self.n_components)
        starts = np.random.rand(self.n_components)
        starts = starts / np.sum(starts)
        ends = np.random.rand(self.n_components)
        self._model = hmm.DenseHMM(dists, edges=edges, starts=starts, ends=ends, max_iter=1)

    def fit_data(self, data):
        """
        # need to add the tags in here
        print("running fit")
        cache_filename = os.path.join(self.cache_dir, 'model.pkl')
        # Check if the pickle file exists
        if os.path.isfile(cache_filename):
            print("loading pickled model")
            # If the file exists, load the variable from the file
            with open(cache_filename, 'rb') as file:
                self._model: hmm.DenseHMM = pickle.load(file)
                # self._model = pickle.load(file)

        else:
            # data = self.push_data_to_gpu(data)
            self._model.fit(X=self.convert_data(data))
            with open(cache_filename, 'wb') as file:
                pickle.dump(self._model, file)
        """
        self._model.fit(X=self.convert_data(data))

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

    def convert_data(self, data):
        data = [torch.from_numpy(arr).reshape(-1, 1) for arr in data]
        return data

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
