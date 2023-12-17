from Config.Config import pipeline_config
import numpy as np


class pipeline:

    def __init__(self, reader, omitter, config: pipeline_config, **kwargs):
        self.reader = reader
        self.omitter = omitter
        self._config = config
        self._transmat_list = []
        self._startprob_list = []
        self._means_list = []

    def fit(self):
        raise NotImplementedError("Must use an implementation of pipeline")

    @property
    def transmat_list(self):
        return self._transmat_list

    @property
    def transmat(self):
        try:
            return self._transmat_list[-1] / np.sum(self._transmat_list[-1], axis=1,
                                                    keepdims=True)  # return the last of the transition matrices
        except AttributeError as e:
            print(f"Model not initialized with fit, exception was raised: {e}")

    @property
    def means_list(self):
        return self._means_list

    @property
    def means(self):
        try:
            return self._means_list[-1] # return the last of the means matrices
        except AttributeError as e:
            print(f"Model not initialized with fit, exception was raised: {e}")

    @property
    def startprob_list(self):
        return self._startprob_list

    @property
    def startprob(self):
        try:
            return self._startprob_list[-1]  # return the last of the starting probability
        except AttributeError as e:
            print(f"Model not initialized with fit, exception was raised: {e}")

    def __str__(self):
        return str(self._config) + str(self.omitter) + str(self.reader)
