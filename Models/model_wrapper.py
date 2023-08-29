
class model_wrapper:

    def __init__(self):
        raise NotImplementedError("Must use an implementation for model wrapper")

    def fit(self,data):
        raise NotImplementedError("Must use an implementation for model wrapper")

    @property
    def transmat(self):
        return None

    @property
    def startprob(self):
        return None

