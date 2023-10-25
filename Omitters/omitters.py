from Omitters.utils import *
from Config.Config import base_omitter_config, bernoulli_omitter_config


class base_omitter:
    def __init__(self, config: base_omitter_config, **kwargs):
        self._config = config

    def omit(self, data):
        return data, None

    def __str__(self):
        return str(self._config)


class bernoulli_omitter(base_omitter):
    def __init__(self, config: bernoulli_omitter_config, **kwargs):
        super().__init__(config, **kwargs)

    # TODO: check if this still works with new data format.
    # TODO: Also, make sure that all the readers are outputting the same format(torch or numpy?)
    def omit(self, data):
        return bernoulli_experiments(self._config.prob_of_observation, data)
