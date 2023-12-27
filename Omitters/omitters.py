from Omitters.utils import *
from Config.Config import base_omitter_config, bernoulli_omitter_config


class base_omitter:
    def __init__(self, config: base_omitter_config):
        self._config = config

    def omit(self, data):
        return data, None

    def __str__(self):
        return str(self._config)


class bernoulli_omitter(base_omitter):
    def __init__(self, config: bernoulli_omitter_config):
        super().__init__(config)

    # TODO: check if this works with all datasets
    def omit(self, data):
        return bernoulli_experiments(self._config.prob_of_observation, data)


class geometric_omitter(base_omitter):
    def __init__(self, config: bernoulli_omitter_config):
        super().__init__(config)

    # TODO: check if this works with all datasets
    def omit(self, data):
        return geometric_experiments(self._config.prob_of_observation, data)

class consecutive_bernoulli_omitter(base_omitter):
    def __init__(self, config: bernoulli_omitter_config):
        super().__init__(config)

    # TODO: check if this works with all datasets
    def omit(self, data):
        return consecutive_bernoulli_experiments(self._config.prob_of_observation, data)

class markov_chain_omitter(base_omitter):
    def __init__(self, config: bernoulli_omitter_config):
        super().__init__(config)

    # TODO: check if this works with all datasets
    def omit(self, data):
        return markov_chain_experiments(self._config.prob_of_observation, data)