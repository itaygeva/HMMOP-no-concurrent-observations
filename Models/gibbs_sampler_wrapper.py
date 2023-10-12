import torch

from Models.model_wrapper import model_wrapper
from who_cell.models import gibbs_sampler
import numpy as np
import random


class gibbs_sampler_wrapper(model_wrapper):
    def __init__(self, n_components, n_iter):
        super().__init__(n_components, n_iter)
        self.model_ = gibbs_sampler.GibbsSampler(length_of_chain=10)

    def fit(self, data):
        data, known_mues, sigmas, start_probs, sentences_lengths = self.generate_initial_parameters(data)
        all_states, all_observations_sum, all_sampled_transitions, all_mues, all_ws, all_transitions, all_omitting_probs \
            = self.model_.sample(data, start_probs, known_mues, sigmas, self.n_iter, N=sentences_lengths)

        transmat = self.create_transmat(all_transitions)
        self.model_.transmat_ = transmat

    def generate_initial_parameters(self, data):
        data = [sentence.numpy() for sentence in data]
        sentences_lengths = [sentence.shape[0] for sentence in data]

        known_mues = None

        sigmas = [random.uniform(1, 2) for i in range(self.n_components)]
        sigmas_dict = {str((sigma, i)): sigma for i, sigma in enumerate(sigmas)}
        start_probs = {state: random.random() for state in sigmas_dict.keys()}
        sum_start_probs = sum(start_probs.values())
        start_probs = {state: prob / sum_start_probs for state, prob in start_probs.items()}

        data = self.prepare_data(data)

        return data, known_mues, sigmas_dict, start_probs, sentences_lengths

    def prepare_data(self, data):
        return [np.squeeze(sentence) for sentence in data]

    def create_transmat(self, all_transitions):
        last_transition_matrix = all_transitions[-1]
        transmat = torch.zeros(self.n_components, self.n_components)
        for index_start in last_transition_matrix.keys():
            if index_start != 'end' and index_start != 'start':

                for index_end in last_transition_matrix[index_start].keys():
                    if index_end != 'end' and index_end != 'start':
                        transmat[int(eval(index_start)[1])][int(eval(index_end)[1])] = \
                        last_transition_matrix[index_start][index_end]
        return transmat

    @property
    def transmat(self):
        return self.model_.transmat_

    @property
    def startprob(self):
        pass
