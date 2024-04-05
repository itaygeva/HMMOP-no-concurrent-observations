import numpy as np
from Config.Config import Config


class base_reader:
    # This class is the parent of all reader and handles the basic query methods they all share

    def __init__(self, config: Config):
        self._config = config
        self.dataset = {'sentences': [], 'tags': [], 'lengths': []}
        self.emission_prob = []

    def get_obs(self):
        """
        :return: returns the observations as a list of sentences. each sentence is a np.array of size (n_obs,n_features)
        """
        return self.dataset['sentences']

    def get_tags(self):
        """
        :return: returns the tags as a list of tag sentences. each tag sentence is a np.array of size(n_obs)
        """
        return self.dataset['tags']

    def get_lengths(self):
        """
        :return: returns a list of the sentence lengths.
        """
        return self.dataset['lengths']

    def get_n_features(self):
        """
        :return: returns the number of features per observation.
        """
        return self._config.n_features

    def get_n_components(self):
        """
        :return: returns the number of states in the HMM model.
        """
        return self._config.n_components

    def get_if_tagged(self):
        """
        :return: returns a boolean whether  the dataset is tagged or not.
        """
        return self._config.is_tagged

    def get_emission_prob(self):
        """
        :return: returns the emission probability matrix if it exists (tagged data)
        """
        if self._config.is_tagged:
            return self.emission_prob
        else:
            return None

    def __str__(self):
        return str(self._config)
