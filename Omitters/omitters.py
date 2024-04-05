from Omitters.utils import *
from Config.Config import base_omitter_config, bernoulli_omitter_config


class base_omitter:
    def __init__(self, config: base_omitter_config):
        self._config = config

    def omit(self, data):
        """
        Passes the data through, without omissions
        :param data: given data from a reader, a list of np.arrays of size (n_components, n_features)
        :return: the omitted data, and a list of the indexes of omissions (in this case None).
        """
        return data, None

    def __str__(self):
        return str(self._config)


class bernoulli_omitter(base_omitter):
    def __init__(self, config: bernoulli_omitter_config):
        super().__init__(config)

    def omit(self, data):
        """
        Omits the data using bernoulli distribtion, where each emission is seen according to a bernoulli trial.
        :param data: given data from a reader, a list of np.arrays of size (n_components, n_features)
        :return: the omitted data, and a list of the indexes of omissions.
        """
        return bernoulli_experiments(self._config.prob_of_observation, data)


class geometric_omitter(base_omitter):
    def __init__(self, config: bernoulli_omitter_config):
        super().__init__(config)

    def omit(self, data):
        """
        Omits the data using geometric distribtion, where the step size between seen emissions is picked each time from a geometric distribution.
        This would be the same as bernoulli, but we use the definition of geometric, where the lowest number is 1. This ensures non-consecutive emissions.
        :param data: given data from a reader, a list of np.arrays of size (n_components, n_features)
        :return: the omitted data, and a list of the indexes of omissions.
        """
        return geometric_experiments(self._config.prob_of_observation, data)


class consecutive_bernoulli_omitter(base_omitter):
    def __init__(self, config: bernoulli_omitter_config):
        super().__init__(config)

    def omit(self, data):
        """
        Omits the data using bernoulli distribtion, where we skip the next emission using a bernoulli trial.
        The lower the bernoulli probability, the more less consecutive emissions we will see.
        :param data: given data from a reader, a list of np.arrays of size (n_components, n_features)
        :return: the omitted data, and a list of the indexes of omissions.
        """
        return consecutive_bernoulli_experiments(self._config.prob_of_observation, data)


class markov_chain_omitter(base_omitter):
    def __init__(self, config: bernoulli_omitter_config):
        super().__init__(config)

    def omit(self, data):
        """
        Omits the data using a markov chain. We run a markov chain with 2 states, emit or omit.
        :param data: given data from a reader, a list of np.arrays of size (n_components, n_features)
        :return: the omitted data, and a list of the indexes of omissions.
        """
        return markov_chain_experiments(self._config.prob_of_observation, data)



class uniform_skips_omitter(base_omitter):
    def __init__(self, config: bernoulli_omitter_config):
        super().__init__(config)

    def omit(self, data):
        """
        Omits the data using uniform distribution, where the step size between seen emissions is picked each time from a uniform distribution.
        :param data: given data from a reader, a list of np.arrays of size (n_components, n_features)
        :return: the omitted data, and a list of the indexes of omissions.
        """
        return uniform_skips_experiment(self._config.n_skips, data)
