import numpy as np
from hmmlearn import hmm

from Config.Config import *
from Pipelines.pipeline import pipeline


class hmmlearn_pipeline(pipeline):

    def __init__(self, reader, omitter, config: hmmlearn_pipeline_config):
        super().__init__(reader, omitter, config)

    def _iter_fit(self, data, n_iter):
        try:
            self.ndim = data[0].shape[1]
        except Exception as e:
            print(f"Incorrect data passed for fit. Raised exception {e}")
        else:
            self.create_model()  # create model based on the dist type
            sentences, lengths = self.convert_data_to_hmmlearn_format(data)
            self._model.fit(sentences, lengths)
            return self._model.transmat_, self._model.startprob_

    def fit(self):
        data, ws = self.omitter.omit(self.reader.get_obs())
        for i in range(1, self._config.n_iter + 1):
            transmat, start_prob = self._iter_fit(data, i)
            self._transmat_list.append(transmat)
            self._startprob_list.append(start_prob)

    def convert_data_to_hmmlearn_format(self, data):
        """
        converts the data to the format that hmmlearn expects in fit
        :param data: the data to convert - list of numpy_array(shape=(n_obs,n_features))
        :return: the converted data - numpy_array(shape=(n_features,n_total_obs))
        """
        if self.ndim == 1:
            data_squeezed = [np.squeeze(sentence) for sentence in data]
            data_hmmlearn_formatted = np.hstack(data_squeezed)
            data_hmmlearn_formatted = data_hmmlearn_formatted.reshape(-1, 1)
        else:  # for the case of multivariate hmm
            data_hmmlearn_formatted = np.transpose(np.vstack(data))
        sentences_length = [int(sentence.shape[0]) for sentence in data]
        return data_hmmlearn_formatted, sentences_length

    # This method should not be used in the end. Just a patch for now
    @staticmethod
    def convert_hmmlearn_format_to_data(hmmlearn_data, lengths):
        """
        Converts from the format that hmmlearn expects in fit to our data format.
        :param hmmlearn_data: the data in hmmlearn format to convert - numpy_array(shape=(n_features,n_total_obs))
        :param lengths: the lengths of the sentences
        :return: the converted data - list of numpy_array(shape=(n_obs,n_features))
        """
        data = []
        start_idx = 0

        for length in lengths:
            end_idx = start_idx + length
            sentence = hmmlearn_data[start_idx:end_idx]
            data.append(sentence)
            start_idx = end_idx

        return data

    def create_model(self):
        """
        Creates the model based on the distribution type. Currently, supports - Gaussian, Categorical.
        :return:
        """
        if self._config.distribution == 'Categorical':
            self._model = hmm.CategoricalHMM(n_components=self._config.n_components, n_iter=self._config.n_iter)
        elif self._config.distribution == 'Gaussian':
            self._model = hmm.GaussianHMM(n_components=self._config.n_components, n_iter=self._config.n_iter)
        else:
            raise NotImplementedError(f"No model implemented for {self._config.distribution} distribution")
        pass
