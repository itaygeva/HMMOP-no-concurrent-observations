import torch

from Models.model_wrapper import model_wrapper
from who_cell.models import gibbs_sampler
import numpy as np
import random


class gibbs_sampler_wrapper(model_wrapper):
    def __init__(self, n_components, n_iter):
        """
        Currently supports only Gaussian distributions
        :param n_components: the number of states the HMM has
        :param n_iter: the number of iterations to have the fit do
        """
        super().__init__(n_components, n_iter)
        self.model_ = gibbs_sampler.GibbsSampler(length_of_chain=10)

    def fit(self, data):
        """
        Fits the gibbs-sampler based HMM model to the given data.
        Finds the model parameters such as the transition matrix.
        :param data: list of hidden markovian sentences. Currently, supports gaussian emissions,
         with single feature sentences.
        """
        data, known_mues, sigmas, start_probs, sentences_lengths = self.generate_initial_parameters(data)
        all_states, all_observations_sum, all_sampled_transitions, all_mues, all_ws, all_transitions, all_omitting_probs \
            = self.model_.sample(data, start_probs, known_mues, sigmas, self.n_iter, N=sentences_lengths)

        self._transmat_list = self.create_transmat_list(all_transitions)
        # TODO: Implement startprob
        self._startprob_list = None

    def generate_initial_parameters(self, data):
        """
        Generates random initial parameters for the gibbs sampler.
        :param data: list of hidden markovian sentences. Currently, supports gaussian emissions,
        :return: return the initial parameters for the gibbs sampler:
        data, known_mues, sigmas_dict, start_probs, sentences_lengths
        """
        data = [sentence.numpy() for sentence in data]
        sentences_lengths = [sentence.shape[0] for sentence in data]

        known_mues = None

        sigmas = [random.uniform(1, 2) for i in range(self.n_components)]
        sigmas_dict = {str((sigma, i)): sigma for i, sigma in enumerate(sigmas)}
        start_probs = {state: random.random() for state in sigmas_dict.keys()}
        sum_start_probs = sum(start_probs.values())
        start_probs = {state: prob / sum_start_probs for state, prob in start_probs.items()}

        data = [np.squeeze(sentence) for sentence in data]

        return data, known_mues, sigmas_dict, start_probs, sentences_lengths

    def create_transmat_list(self, all_transitions):
        """
        Creates a transition matrix of our format (torch tenor of shape (n_states, n_states))
        from the last element of the gibbs_sampler all_transitions parameter
        :param all_transitions:  A list of n_iter length of the dictionary of found edges weights.
        :return: The final transition matrix
        """
        transmat_list = []
        for transition_matrix in all_transitions:
            transmat = torch.zeros(self.n_components, self.n_components)
            for index_start in transition_matrix.keys():
                if index_start != 'end' and index_start != 'start':
                    for index_end in transition_matrix[index_start].keys():
                        if index_end != 'end' and index_end != 'start':
                            transmat[int(eval(index_start)[1])][int(eval(index_end)[1])] = \
                                transition_matrix[index_start][index_end]
            transmat_list.append(transmat)
        return transmat_list

