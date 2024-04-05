import random

import numpy as np

from Config.Config import *
from Pipelines.pipeline import pipeline
from who_cell.models import gibbs_sampler


class gibbs_sampler_pipeline(pipeline):
    def __init__(self, reader, omitter, config: gibbs_sampler_pipeline_config):
        """
        :param reader: the initialized reader
        :param omitter: the initialized reader
        :param config: the config
        """
        super().__init__(reader, omitter, config)

        # the gibbs sampler needs to know the length of the sentence.
        # this is the length of chain. we give a list of the length in the fit
        self.model_ = gibbs_sampler.GibbsSampler(length_of_chain=None)

    def fit(self):
        """
        Fits the model on the omitted data, and creates the transmat, startprob and means list.
        """
        data, ws = self.omitter.omit(self.reader.get_obs())
        data, known_mues, sigmas, start_probs, sentences_lengths, ws = self.generate_initial_parameters(data, ws)
        all_states, all_observations_sum, all_sampled_transitions, all_mues, all_ws, all_transitions, all_omitting_probs \
            = self.model_.sample_known_W(data, start_probs, known_mues, sigmas, self._config.n_iter, ws,
                                         N=sentences_lengths)
        self._means_list = self.create_means_list(all_mues)
        self._transmat_list = self.create_transmat_list(all_transitions)
        self._startprob_list = self.create_startprob_list(all_transitions)

    def generate_initial_parameters(self, data, ws):
        """
        Generates the initial parameteres of the model using information on the original data from the reader
        and information on the known omission locations from the omission. Also prepares the data.
        :param data: the omitted data to prepare
        :param ws: the seen emission locations
        :return: data - the prepared data, mues_dict - the dictionary of state-mu , sigmas_dict - the dictionary of state-sigma,
        start_probs - the starting probability, ws - the seen emission locations
        """
        sentences_lengths = self.reader.dataset['lengths']
        if ws is None:
            ws = [list(np.arange(length)) for length in sentences_lengths]

        known_mues = self.reader.means

        sigmas = self.reader.covs
        sigmas_dict = {str((sigma, i)): sigma for i, sigma in enumerate(sigmas)}
        mues_dict = {str((sigma, i)): known_mues[i] for i, sigma in enumerate(sigmas)}
        start_probs = {state: random.random() for state in sigmas_dict.keys()}
        sum_start_probs = sum(start_probs.values())
        start_probs = {state: prob / sum_start_probs for state, prob in start_probs.items()}

        data = [np.squeeze(sentence) for sentence in data]  # is this needed?

        return data, mues_dict, sigmas_dict, start_probs, sentences_lengths, ws

    def create_startprob_list(self, all_transitions):
        """
        Creates a list of starting probabilities from the gibbs_sampler all_transitions parameter
        :param all_transitions: A list of n_iter length of the dictionary of found edges weights.
        :return: a list of the fitted starting probabilities
        """
        # # TODO: There seems to maybe a bug in Benny's code, where after the first iteration (the initialization),
        #  the start prob is zero for all states. We can see that this is because in the walk, we don't start with the 'start' state, and therefore we don't count the transitions from it.
        #   Also notice than he returns iter+1

        startprob_list = []
        for transition_matrix in all_transitions:
            startprob = np.zeros(self._config.n_components)
            for index_end in transition_matrix['start'].keys():
                if index_end != 'end' and index_end != 'start':
                    startprob[int(eval(index_end)[1])] = \
                        transition_matrix['start'][index_end]
            startprob = startprob / np.sum(startprob)
            startprob_list.append(startprob)
        return startprob_list

    def create_transmat_list(self, all_transitions):
        """
        Creates a list of transition matrices of our format list(torch tenor of shape (n_states, n_states))
        from the gibbs_sampler all_transitions parameter
        :param all_transitions: A list of n_iter length of the dictionary of found edges weights.
        :return: a list of the fitted transition matrices
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

    def create_means_list(self, all_mues):
        return [np.array(list(mues.values())) for mues in all_mues]
