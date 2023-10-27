import numpy as np
import pomegranate
import torch
from pomegranate.hmm import DenseHMM

from Config.Config import synthetic_reader_config
from Data.Readers.base_reader import base_reader


class synthetic_reader(base_reader):

    def __init__(self, config: synthetic_reader_config):
        super().__init__(config)
        self._tag_dict = {}
        self._word_dict = {}
        self._transition_mat = torch.zeros([self._config.n_components, self._config.n_components])
        self._end_probs = torch.zeros([self._config.n_components, 1])
        self._start_probs = None
        self._distributions = None
        self._init_hmm_model()
        self._generate_samples()

    def _generate_samples(self):
        self.dataset['sentences'] = self._model.sample(self._config.n_samples)
        self.dataset['sentences'] = [sentence.numpy() for sentence in self.dataset['sentences']]
        self.dataset['lengths'] = [sample.shape[0] for sample in self.dataset['sentences']]
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
        min_mu_gap = 10
        sigma = 1
        mues = [i * min_mu_gap for i in range(self._config.n_components)]
        self._distributions = np.array(
            [pomegranate.distributions.Normal([float(mu)], [float(sigma)], 'diag') for mu in mues])

        # generating transitions probs
        for i in range(self._config.n_components):
            random_line = torch.rand(self._config.n_components + 1)
            random_line = random_line / torch.sum(random_line)
            self._transition_mat[i, :] = random_line[:-1]
            self._end_probs[i] = random_line[-1]

        self._start_probs = torch.rand(self._config.n_components)
        self._start_probs = self._start_probs / torch.sum(self._start_probs)

        self._end_probs = self._end_probs.squeeze() # why is this needed?

        # generating model
        self._model = DenseHMM(self._distributions, self._transition_mat, self._start_probs, self._end_probs)


    def delete_one_word_sentences(self):
        no_one_word_data = [(length, sentences) for (length, sentences) in
                            zip(self.dataset['lengths'], self.dataset['sentences']) if length > 1]
        self.dataset['lengths'], self.dataset['sentences'] = zip(*no_one_word_data)


    @property
    def transmat(self):
        return self._transition_mat

    @property
    def startprob(self):
        return self._start_probs