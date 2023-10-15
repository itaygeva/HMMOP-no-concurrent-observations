from hmmlearn import hmm
import numpy as np
from Models.model_wrapper import model_wrapper


class matrix_wrapper(model_wrapper):
    def __init__(self):
        self._transmat = None
        self._startprob = None

    def fit(self, transmat):
        self._transmat = transmat
        # TODO : Add get startprob to synthetic reader

    def transmat(self):
        return self._transmat

    def startprob(self):
        return self._startprob
