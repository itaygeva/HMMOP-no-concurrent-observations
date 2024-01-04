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

    def generate_near_biased_matrix(self):
        stochastic_matrix = np.empty((self._config.n_components, self._config.n_components))
        for i in range(self._config.n_components):
            for j in range(self._config.n_components):
                # Example: Higher probability for transitions between nearby states
                stochastic_matrix[i, j] = np.exp(-abs(i - j)) / np.sum(np.exp(-np.abs(np.arange(self._config.n_components) - i)))

        # Ensure rows sum to 1
        stochastic_matrix /= stochastic_matrix.sum(axis=1, keepdims=True)
        return np.linalg.matrix_power(stochastic_matrix, self._config.matrix_power)

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
        if self._config.set_temporal:
            self._transition_mat = self.generate_near_biased_matrix()
        elif self._config.transmat is None:
            self._transition_mat = np.random.rand(self._config.n_components, self._config.n_components)
            for line in self._transition_mat:
                line /= np.sum(line)
        else:
            self._transition_mat = self.generate_param_from_config(self._config.transmat)

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
