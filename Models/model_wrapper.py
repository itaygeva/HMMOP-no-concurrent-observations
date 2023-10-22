
class model_wrapper:

    def __init__(self, n_components, n_iter):
        self.n_components = n_components
        self.n_iter = n_iter
        self._transmat_list = []
        self._startprob_list = []

    def fit(self, data):
        raise NotImplementedError("Must use an implementation for model wrapper")

    @property
    def transmat_list(self):
        return self._transmat_list
    @property
    def transmat(self):
        try:
            return self._transmat_list[-1]  # return the last of the transition matrices
        except AttributeError as e:
            print(f"Model not initialized with fit, exception was raised: {e}")

    @property
    def startprob_list(self):
        return self._startprob_list
    @property
    def startprob(self):
        try:
            return self._startprob_list[-1]  # return the last of the transition matrices
        except AttributeError as e:
            print(f"Model not initialized with fit, exception was raised: {e}")




