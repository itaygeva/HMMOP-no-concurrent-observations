from Config.Config import *
from Pipelines.pipeline import pipeline


class matrix_pipeline(pipeline):
    def __init__(self, reader, omitter, config: matrix_pipeline_config):
        """
        :param reader: the initialized reader
        :param omitter: the initialized reader
        :param config: the config
        """
        super().__init__(reader, omitter, config)
        self._transmat = None
        self._startprob = None

    def fit(self):
        """
        Extracts ground truth transition matrix, starting probability and means from the reader
        """
        self._transmat_list = [self.reader.transmat for i in range(self._config.n_iter+1)]
        self._startprob_list = [self.reader.startprob for i in range(self._config.n_iter+1)]
        self._means_list = [self.reader.means for i in range(self._config.n_iter+1)]
