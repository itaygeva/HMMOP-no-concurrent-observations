import numpy as np
import pomegranate
from pomegranate.hmm import DenseHMM
import os

from Config.Config import synthetic_reader_config
from Data.Readers.base_reader import base_reader


class my_synthetic_reader(base_reader):

    def __init__(self, config: synthetic_reader_config):
        super().__init__(config)
        self._tag_dict = {}
        self._word_dict = {}
        self._init_hmm_model()
        self._generate_samples()

    def _normalize_stochastic_matrix(self, matrix):
        for row in matrix:
            row /= np.sum(row)
        return matrix

    def generate_near_biased_matrix(self):
        rows = np.arange(self._config.n_components)
        # Create a 2D array with repeated rows to represent the column indices
        columns = np.tile(rows, (self._config.n_components, 1))
        # Calculate the absolute differences element-wise
        distance_matrix = -1 * np.abs(rows - columns.T)
        stochastic_matrix = np.exp(distance_matrix)

        return self._normalize_stochastic_matrix(stochastic_matrix)

    def generate_n_edges_matrix(self, n_edges):
        stochastic_matrix = np.zeros((self._config.n_components, self._config.n_components))
        for row in stochastic_matrix:
            edges_idx = np.random.choice(self._config.n_components, size=(n_edges), replace=False)
            edges_values = np.random.random(n_edges)
            row[edges_idx] = edges_values

        return self._normalize_stochastic_matrix(stochastic_matrix)

    def _generate_samples(self):
        # %% create markov chain
        self.dataset['lengths'] = [self._config.sentence_length] * self._config.n_samples
        sentences = []
        for sentence_len in self.dataset['lengths']:
            sentence = []
            initial_state = np.random.choice(np.arange(self._config.n_components), p=self._start_prob)
            current_state = initial_state

            for _ in range(sentence_len):
                sentence.append(current_state)
                current_state = np.random.choice(np.arange(self._config.n_components),
                                                 p=self._transition_mat[current_state])
            sentences.append(sentence)

        # %% create observations
        self.dataset['sentences'] = []

        for sentence in sentences:

            observations = []

            for word in sentence:
                observation = np.random.normal(self._mues[word], self._sigmas[word])
                observations.append(observation)
            self.dataset['sentences'].append(np.array(observations))

        self.delete_one_word_sentences()  # this is so we won't get an error in the gibbs sampler

    def _init_hmm_model(self):
        """
        this function initialize the model for the synthetic sample. this functions has two modes:
            _init_hmm_model(self, model_num): used for prying a model that is already in cache
            _init_hmm_model(self, min_mu_gap, sigma): used for creating a new model.
        """
        ## TODO: add configuration for the mus and sigmas and transition and start
        ## TODO: verify that the distribution is created as we want
        # generating distributions
        self.generate_mues()
        self.generate_sigmas()
        self.generate_transmat()
        self.generate_startprobs()

        # generating model

    def delete_one_word_sentences(self):
        no_one_word_data = [(length, sentences) for (length, sentences) in
                            zip(self.dataset['lengths'], self.dataset['sentences']) if length > 1]
        self.dataset['lengths'], self.dataset['sentences'] = map(list, zip(*no_one_word_data))

    def generate_param_from_config(self, param_path):
        with open(os.path.join(self._config.params_dir, param_path), 'r') as file:
            content = file.read()
        return eval(content)

    def generate_mues(self):
        if self._config.mues is None:
            min_mu_gap = 10
            self._mues = [i * min_mu_gap for i in range(self._config.n_components)]
        else:
            self._mues = self.generate_param_from_config(self._config.mues)

    def generate_sigmas(self):
        if self._config.sigma is None:
            self._sigmas = [1 for i in range(self._config.n_components)]
        else:
            self._sigmas = self.generate_param_from_config(self._config.sigma)

    def generate_transmat(self):
        if self._config.transmat_mode == "Near Neighbors":
            self._transition_mat = self.generate_near_biased_matrix()
        elif self._config.transmat_mode == "N Edges":
            self._transition_mat = self.generate_n_edges_matrix(self._config.n_edges)
        elif self._config.transmat_mode == "Config":
            self._transition_mat = self.generate_param_from_config(self._config.transmat)
        else:
            self._transition_mat = np.random.rand(self._config.n_components, self._config.n_components)
            self._transition_mat = self._normalize_stochastic_matrix(self._transition_mat)

        if self._config.set_temporal:
            self._transition_mat = np.linalg.matrix_power(self._transition_mat, self._config.matrix_power)

    def generate_startprobs(self):
        if self._config.startprobs is None:
            self._start_prob = np.random.rand(self._config.n_components)
            self._start_prob = self._start_prob / np.sum(self._start_prob)
        else:
            self._start_prob = self.generate_param_from_config(self._config.startprobs)

    @property
    def transmat(self):
        return self._transition_mat

    @property
    def means(self):
        return self._mues

    @property
    def covs(self):
        return self._sigmas

    @property
    def startprob(self):
        return self._start_prob
