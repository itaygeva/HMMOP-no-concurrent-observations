
class model_wrapper:

    def __init__(self, n_components, n_iter):
        self.n_components = n_components
        self.n_iter = n_iter

    def fit(self,data):
        raise NotImplementedError("Must use an implementation for model wrapper")

    @property
    def transmat(self):
        return None

    @property
    def startprob(self):
        return None

