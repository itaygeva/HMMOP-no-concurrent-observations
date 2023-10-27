from Config.Config import pipeline_config


class pipeline:

    def __init__(self, reader, omitter, config: pipeline_config, **kwargs):
        self.reader = reader
        self.omitter = omitter
        self._config = config
        self._transmat_list = []
        self._startprob_list = []

    def fit(self):
        raise NotImplementedError("Must use an implementation of pipeline")

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

    def __str__(self):
        return str(self._config) + str(self.omitter) + str(self.reader)
