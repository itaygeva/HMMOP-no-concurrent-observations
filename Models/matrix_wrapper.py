from hmmlearn import hmm
import numpy as np
from Models.model_wrapper import model_wrapper


class matrix_wrapper(model_wrapper):
    def __init__(self, n_iter):
        self._transmat = None
        self._startprob = None
        self.n_iter = n_iter

    def fit(self, transmat):
        self._transmat_list = [transmat for i in range(self.n_iter)]
        # TODO : Add get startprob to synthetic reader
        self._startprob = [None for i in range(self.n_iter)]
