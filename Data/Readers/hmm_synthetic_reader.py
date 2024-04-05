import numpy as np
import pomegranate
from hmmlearn import hmm
from Data.Readers.utils import *
from Config.Config import hmm_synthetic_reader_config
from Data.Readers.base_reader import base_reader
import os


class hmm_synthetic_reader(base_reader):
    def __init__(self, config: hmm_synthetic_reader_config):
        super().__init__(config)
        self._tag_dict = {}
        self._word_dict = {}
        self._init_hmm_model()
        self._generate_samples()

    def _init_hmm_model(self):
        self._model = hmm.GaussianHMM(n_components=self._config.n_components, covariance_type="full")
        self.generate_mues()
        self.generate_sigmas()
        self.generate_transmat()
        self.generate_startprobs()

    def generate_param_from_config(self, param_path):
        with open(os.path.join(self._config.params_dir, param_path), 'r') as file:
            content = file.read()
        return eval(content)

    def generate_mues(self):
        if self._config.mues is None:
            min_mu_gap = 10
            mues = np.array([i * min_mu_gap for i in range(self._config.n_components)])
        else:
            mues = np.array(self.generate_param_from_config(self._config.mues))
        if mues.ndim == 1:
            mues = np.expand_dims(mues, axis=1)
        self._model.means_ = mues

    def generate_sigmas(self):
        if self._config.sigma is None:
            sigmas = np.array([1 for i in range(self._config.n_components)])
        else:
            sigmas = np.array(self.generate_param_from_config(self._config.sigma))
        if sigmas.ndim == 1:
            sigmas = np.expand_dims(sigmas, axis=1)
            sigmas = np.expand_dims(sigmas, axis=1)
        self._model.covars_ = sigmas

    def generate_transmat(self):
        if self._config.transmat is None:
            transition_mat = np.random.rand(self._config.n_components, self._config.n_components)
            for line in transition_mat:
                line /= np.sum(line)
        else:
            transition_mat = self.generate_param_from_config(self._config.transmat)
        self._model.transmat_ = transition_mat

    def generate_startprobs(self):
        if self._config.startprobs is None:
            self._model.startprob_ = generate_random_normalized_vector(self._config.n_components)
        else:
            self._model.startprob_ = self.generate_param_from_config(self._config.startprobs)

    def _generate_samples(self):
        for i in range(self._config.n_samples):
            sentence, tags = self._model.sample(self._config.sentence_length)
            self.dataset['sentences'].append(sentence.astype(np.float32))

        self.dataset['lengths'] = [self._config.sentence_length] * self._config.n_samples

    def _generate_samples_old(self):
        X, Z = self._model.sample(self._config.n_samples)
        n_sentences = X.shape[0] // self._config.sentence_length
        max_idx = (X.shape[0] // self._config.sentence_length) * self._config.sentence_length

        self.dataset['sentences'] = np.split(X.astype(np.float32)[:max_idx], n_sentences)
        self.dataset['lengths'] = [self._config.sentence_length] * n_sentences

    def _generate_samples_old_old(self):
        X, Z = self._model.sample(self._config.n_samples)
        self.dataset['sentences'] = np.array_split(X.astype(np.float32), X.shape[0] // self._config.sentence_length)
        self.dataset['lengths'] = [sample.shape[0] for sample in self.dataset['sentences']]
        self.delete_one_word_sentences()  # this is so we won't get an error in the gibbs sampler

    def delete_one_word_sentences(self):
        no_one_word_data = [(length, sentences) for (length, sentences) in
                            zip(self.dataset['lengths'], self.dataset['sentences']) if length > 1]
        self.dataset['lengths'], self.dataset['sentences'] = map(list, zip(*no_one_word_data))

    @property
    def transmat(self):
        return self._model.transmat_

    @property
    def means(self):
        return self._model.means_

    @property
    def covs(self):
        return np.squeeze(self._model.covars_)

    @property
    def startprob(self):
        return self._model.startprob_
