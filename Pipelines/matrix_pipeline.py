from hmmlearn import hmm
import numpy as np
from Pipelines.pipeline import pipeline
from Config.Config import *
from Data.Readers import *
from Omitters import *


## TODO: Change the name wrapper to pipeline. Now that we are working in pipeline
# (holds reader and omitter, maybe groundtruth should be attribute in pipeline, and not a wrapper just for it )
class matrix_pipeline(pipeline):
    def __init__(self, reader, omitter, config: matrix_pipeline_config, **kwargs):
        super().__init__(reader, omitter, config, **kwargs)
        self._transmat = None
        self._startprob = None

    def fit(self):
        transmat = self.reader.transition_mat
        self._transmat_list = [transmat for i in range(self._config.n_iter)]
        # TODO : Add get startprob to synthetic reader
        self._startprob = [None for i in range(self._config.n_iter)]
