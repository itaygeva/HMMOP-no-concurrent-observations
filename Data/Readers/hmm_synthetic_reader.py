import numpy as np
import pomegranate
from hmmlearn import hmm
from pomegranate.hmm import DenseHMM
from Data.Readers.utils import *
from Config.Config import hmm_synthetic_reader_config
from Data.Readers.base_reader import base_reader


class hmm_synthetic_reader(base_reader):
    def __init__(self, config: hmm_synthetic_reader_config):
        super().__init__(config)
        self._tag_dict = {}
        self._word_dict = {}
        self._init_hmm_model()
        self._generate_samples()

    def _init_hmm_model(self):
        self._model = hmm.GaussianHMM(n_components=self._config.n_components, n_iter=20, covariance_type="full")
        self._model.startprob_ = generate_random_normalized_vector(self._config.n_components)
        self._model.transmat_ = generate_random_normalized_matrix((self._config.n_components, self._config.n_components))
        self._model.means_ = np.expand_dims(np.random.uniform(1, 10, size=self._config.n_components), axis=1)
        self._model.covars_ = np.tile(0.01*np.identity(1), (self._config.n_components, 1, 1))


    def _generate_samples(self):
        X, Z = self._model.sample(self._config.n_samples)
        self.dataset['sentences'] = np.array_split(X.astype(np.float32), X.shape[0]//10)
        self.dataset['lengths'] = [sample.shape[0] for sample in self.dataset['sentences']]
        self.delete_one_word_sentences()  # this is so we won't get an error in the gibbs sampler

    def delete_one_word_sentences(self):
        no_one_word_data = [(length, sentences) for (length, sentences) in
                            zip(self.dataset['lengths'], self.dataset['sentences']) if length > 1]
        self.dataset['lengths'], self.dataset['sentences'] = zip(*no_one_word_data)

    @property
    def transmat(self):
        return self._model.transmat_

    @property
    def means(self):
        return self._model.means_

    @property
    def startprob(self):
        return self._model.startprob_
