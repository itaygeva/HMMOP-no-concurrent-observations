import os

import pomegranate.hmm as hmm
import pomegranate.distributions as distributions
import random
import numpy as np
from Models.model_wrapper import model_wrapper
import pickle
import inspect
import torch


class pome_wrapper(model_wrapper):
    def __init__(self, n_components, n_iter, emission_prob, freeze_distributions=False):
        super().__init__(n_components, n_iter)
        self._model = None
        self.emission_prob = emission_prob
        self.freeze_distributions = freeze_distributions

        print("creating model")
        # create model
        self.generate_initial_model()

        data_dir = os.path.dirname(inspect.getfile(model_wrapper))
        self.cache_dir = os.path.join(data_dir, 'Cache')

    def generate_initial_model(self):
        dists = [distributions.Categorical(np.atleast_2d(dist), frozen=self.freeze_distributions) for dist in self.emission_prob]
        edges = np.random.rand(self.n_components, self.n_components)
        starts = np.random.rand(self.n_components)
        starts = starts/np.sum(starts)
        ends = np.random.rand(self.n_components)
        self._model = hmm.DenseHMM(dists, edges=edges, starts=starts, ends=ends, max_iter=1)

    def fit(self, data):
        # need to add the tags in here
        print("running fit")
        cache_filename = os.path.join(self.cache_dir, 'model.pkl')
        # Check if the pickle file exists
        if os.path.isfile(cache_filename):
            # If the file exists, load the variable from the file
            with open(cache_filename, 'rb') as file:
                self._model = pickle.load(file)

        else:
            # data = self.push_data_to_gpu(data)
            self._model = hmm.DenseHMM()
            self._model.fit(X=self.convert_data(data))
            with open(cache_filename, 'wb') as file:
                pickle.dump(self._model, file)

    @property
    def transmat(self):
        try:
            return self._model.edges
        except AttributeError as e:
            print(f"Model not initialized with fit, exception was raised: {e}")

    @property
    def startprob(self):
        try:
            return self._model.starts
        except AttributeError as e:
            print(f"Model not initialized with fit, exception was raised: {e}")

    @property
    def emissionprob(self):
        return self.create_emissionprob()

    def create_emissionprob(self):
        self._model = hmm.DenseHMM()
        emission_prob = []
        for prob in self._model.distributions:
            emission_prob.append(np.array(prob.probs))

        return np.vstack(emission_prob)

    def push_data_to_gpu(self, data):
        tensors = [torch.tensor(sentence).float().cuda() for sentence in data]
        return tensors

    def convert_data(self, data):
        data = [torch.from_numpy(arr).reshape(-1,1) for arr in data]
        print(data[0])
        return data





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
