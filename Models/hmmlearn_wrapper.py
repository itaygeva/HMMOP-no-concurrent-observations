from hmmlearn import hmm
import numpy as np
from Models.model_wrapper import model_wrapper


class hmmlearn_wrapper(model_wrapper):

    def __init__(self, n_components, n_iter, emission_prob):
        # we only want to find the startprob_ - 's' and the transmat_ 't
        # self._model = hmm.CategoricalHMM(n_components=num_states, n_iter=num_iter, init_params='st')
        super().__init__(n_components, n_iter)
        self._model = None
        self.emission_prob = emission_prob

        # self._model.emissionprob_= emission_prob

    def fit(self, data):
        self._model = hmm.CategoricalHMM(n_components=self.n_components, n_iter=self.n_iter)
        try:
            self.ndim = data[0].ndim
        except Exception as e:
            print(f"Incorrect data passed for fit. Raised exception {e}")
        else:
            sentences, lengths = self.convert_data_to_hmmlearn_format(data)
            if self.ndim == 1:
                sentences = sentences.reshape(-1, 1)
            self._model.fit(sentences, lengths)

    def convert_data_to_hmmlearn_format(self, data):
        """
        converts the data to the format that hmmlearn expects in fit
        :param data: the data to convert - list of numpy_array(shape=(n_obs,n_features))
        :return: the converted data - numpy_array(shape=(n_features,n_total_obs))
        """
        ndim = data[0].ndim
        if self.ndim == 1:
            data_hmmlearn_formatted = np.hstack(data)
        else:  # for the case of multivariate hmm
            data_hmmlearn_formatted = np.transpose(np.vstack(data))
        sentences_length = [int(sentence.shape[0]) for sentence in data]
        return data_hmmlearn_formatted, sentences_length

    @property
    def transmat(self):
        try:
            return self._model.transmat_
        except AttributeError as e:
            print(f"Model not initialized with fit, exception was raised: {e}")

    @property
    def startprob(self):
        try:
            return self._model.startprob_
        except AttributeError as e:
            print(f"Model not initialized with fit, exception was raised: {e}")

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
