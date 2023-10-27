import random

import numpy as np

from Config.Config import *
from Pipelines.pipeline import pipeline
from who_cell.models import gibbs_sampler


class gibbs_sampler_pipeline(pipeline):
    def __init__(self, reader, omitter, config: gibbs_sampler_pipeline_config):
        """
        Currently supports only Gaussian distributions
        :param n_components: the number of states the HMM has
        :param n_iter: the number of iterations to have the fit do
        """
        super().__init__(reader, omitter, config)

        # the gibbs sampler needs to know the length of the sentence.
        # this is the length of chain. we give a list of the length in the fit
        self.model_ = gibbs_sampler.GibbsSampler(length_of_chain=None)

    def fit(self):
        """
        Fits the gibbs-sampler based HMM model to the given data.
        Finds the model parameters such as the transition matrix.
        :param data: list of hidden markovian sentences. Currently, supports gaussian emissions,
         with single feature sentences.
        """
        data, ws = self.omitter.omit(self.reader.get_obs())
        data, known_mues, sigmas, start_probs, sentences_lengths = self.generate_initial_parameters(data)
        all_states, all_observations_sum, all_sampled_transitions, all_mues, all_ws, all_transitions, all_omitting_probs \
            = self.model_.sample(data, start_probs, known_mues, sigmas, self._config.n_iter, N=sentences_lengths)

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
        sentences_lengths = [sentence.shape[0] for sentence in data]

        known_mues = None

        sigmas = [random.uniform(1, 2) for i in range(self._config.n_components)]
        sigmas_dict = {str((sigma, i)): sigma for i, sigma in enumerate(sigmas)}
        start_probs = {state: random.random() for state in sigmas_dict.keys()}
        sum_start_probs = sum(start_probs.values())
        start_probs = {state: prob / sum_start_probs for state, prob in start_probs.items()}

        data = [np.squeeze(sentence) for sentence in data] # is this needed?

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
            transmat = np.zeros((self._config.n_components, self._config.n_components))
            for index_start in transition_matrix.keys():
                if index_start != 'end' and index_start != 'start':
                    for index_end in transition_matrix[index_start].keys():
                        if index_end != 'end' and index_end != 'start':
                            transmat[int(eval(index_start)[1])][int(eval(index_end)[1])] = \
                                transition_matrix[index_start][index_end]
                    transmat[int(eval(index_start)[1])] = \
                        transmat[int(eval(index_start)[1])] / np.sum(transmat[int(eval(index_start)[1])])
            transmat_list.append(transmat)
        return transmat_list

