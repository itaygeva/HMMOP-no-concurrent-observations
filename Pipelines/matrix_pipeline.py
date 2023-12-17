from Config.Config import *
from Pipelines.pipeline import pipeline


## TODO: Change the name wrapper to pipeline. Now that we are working in pipeline
# (holds reader and omitter, maybe groundtruth should be attribute in pipeline, and not a wrapper just for it )
class matrix_pipeline(pipeline):
    def __init__(self, reader, omitter, config: matrix_pipeline_config):
        super().__init__(reader, omitter, config)
        self._transmat = None
        self._startprob = None

    def fit(self):
        self._transmat_list = [self.reader.transmat for i in range(self._config.n_iter)]
        self._startprob_list = [self.reader.startprob for i in range(self._config.n_iter)]
        self._means_list = [self.reader.means for i in range(self._config.n_iter)]
