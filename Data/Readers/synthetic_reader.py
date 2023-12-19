import numpy as np
import pomegranate
import torch
from pomegranate.hmm import DenseHMM
import os

from Config.Config import synthetic_reader_config
from Data.Readers.base_reader import base_reader


class synthetic_reader(base_reader):

    def __init__(self, config: synthetic_reader_config):
        super().__init__(config)
        self._tag_dict = {}
        self._word_dict = {}
        self._transition_mat = torch.zeros([self._config.n_components, self._config.n_components])
        self._end_probs = torch.zeros([self._config.n_components, 1])
        self._start_prob = None
        self._distributions = None
        self._init_hmm_model()
        self._generate_samples()

    def _generate_samples(self):
        self.dataset['sentences'] = self._model.sample(self._config.n_samples)
        self.dataset['sentences'] = [sentence.numpy() for sentence in self.dataset['sentences']]
        self.dataset['lengths'] = [sample.shape[0] for sample in self.dataset['sentences']]
        self.delete_one_word_sentences()  # this is so we won't get an error in the gibbs sampler

    def _init_hmm_model_from_config(self):
        self._mues = self.generate_param_from_config(self._config.mues)
        self._sigmas = self.generate_param_from_config(self._config.sigma)
        self._distributions = np.array(
            [pomegranate.distributions.Normal([float(mu)], [float(self._sigmas[i])], 'diag') for i, mu in
             enumerate(self._mues)])
        self._transition_mat = self.generate_param_from_config(self._config.transmat)
        self._start_prob = self.generate_param_from_config(self._config.startprob)
        self._end_probs = self.generate_param_from_config(self._config.endprobs)

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
        self.generate_transmat_and_endprobs()
        self.generate_startprobs()

        # generating model
        self._distributions = np.array(
            [pomegranate.distributions.Normal([float(mu)], [float(self._sigmas[i])], 'diag') for i, mu in
             enumerate(self._mues)])
        self._model = DenseHMM(self._distributions, self._transition_mat, self._start_prob, self._end_probs)

    def delete_one_word_sentences(self):
        no_one_word_data = [(length, sentences) for (length, sentences) in
                            zip(self.dataset['lengths'], self.dataset['sentences']) if length > 1]
        self.dataset['lengths'], self.dataset['sentences'] = zip(*no_one_word_data)

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

    def generate_transmat_and_endprobs(self):
        if self._config.transmat is None or self._config.endprobs is None:
            for i in range(self._config.n_components):
                random_line = torch.rand(self._config.n_components + 1)
                random_line = random_line / torch.sum(random_line)
                self._transition_mat[i, :] = random_line[:-1]
                self._end_probs[i] = random_line[-1]

            self._end_probs = self._end_probs.squeeze()  # why is this needed?
        else:
            self._transition_mat = self.generate_param_from_config(self._config.transmat)
            self._end_probs = self.generate_param_from_config(self._config.endprobs)

    def generate_startprobs(self):
        if self._config.startprobs is None:
            self._start_prob = torch.rand(self._config.n_components)
            self._start_prob = self._start_prob / torch.sum(self._start_prob)
        else:
            self._start_prob = self.generate_param_from_config(self._config.startprobs)

    @property
    def transmat(self):
        return self._transition_mat.numpy()

    @property
    def means(self):
        return self._mues

    @property
    def covs(self):
        return self._sigmas

    @property
    def startprob(self):
        return self._start_prob.numpy()
