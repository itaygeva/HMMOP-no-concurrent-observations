from hmmlearn import hmm
import numpy as np
from Models.model_wrapper import model_wrapper


class hmmlearn_wrapper(model_wrapper):

    def __init__(self, num_states, num_iter):
        self._model = hmm.GaussianHMM(n_components=num_states, n_iter=num_iter)

    def fit(self, data):
        sentences, lengths = self.convert_data_to_hmmlearn_format(data)
        self._model.fit(sentences, lengths)

    def convert_data_to_hmmlearn_format(self, data):
        """
        converts the data to the format that hmmlearn expects in fit
        :param data: the data to convert - list of numpy_array(shape=(n_obs,n_features))
        :return: the converted data - numpy_array(shape=(n_features,n_total_obs))
        """
        data_hmmlearn_formatted = np.transpose(np.vstack(data))
        sentences_length = [sentence.shape[0] for sentence in data].astype(int)
        return data_hmmlearn_formatted, sentences_length

    @property
    def transmat(self):
        return self._model.transmat_

    @property
    def startprob(self):
        return self._model.startprob_

# This method should not be used in the end. Just a patch for now
    @staticmethod
    def convert_hmmlearn_format_to_data(hmmlearn_data, lengths):
        """
        converts from the format that hmmlearn expects in fit to our data format
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
