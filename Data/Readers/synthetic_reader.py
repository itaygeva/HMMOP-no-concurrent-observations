from Data.Readers.base_reader import BaseReader
import sys
import os
from Data.Readers.utils import Utils
from itertools import chain
from collections import namedtuple, OrderedDict
import snowballstemmer
import string
import inspect
import numpy as np
import pickle
from scipy.stats import binom
import pprint
import pomegranate

inspect.getfile(BaseReader)


class SyntheticReader(BaseReader):

    def __init__(self, n_states, sigma):
        super().__init__('')
        self.n_features = 1
        self.is_tagged = True
        self.tag_dict = {}
        self.word_dict = {}
        self.n_states = n_states
        self.distributions, self.states = self._calc_state_distributions(sigma)

    def generate_samples(self, sentence_len):
        pass

    def _calc_state_distributions(self, sigma):
        N = self.n_states
        cache_filename = os.path.join(self.cache_dir, 'synthetic_distributions_n_states_' + self.n_states + 'sigma_'
                                      + sigma + '.py')

        p_prob_of_observation = 0.5

        mues = list(range(N)) * 10
        sigmas = [1 for i in range(N)]

        binom_dist = binom(N, p_prob_of_observation)
        k = binom_dist.rvs()

        full_sample = list(range(N))  # [dist.sample() for dist in distrbutions]
        partial_sample = [full_sample[i] for i in sorted(np.random.choice(range(N), k, replace=False))]

        # Check if the pickle file exists
        if os.path.isfile(cache_filename):
            # If the file exists, load the variable from the file
            with open(cache_filename, 'rb') as file:
                distributions = pickle.load(file)
        else:
            distributions = [pomegranate.distributions.NormalDistribution(mues[i], sigmas[i]) for i in range(N)]
            # If the file doesn't exist, create the variable and save it to the file
            with open(cache_filename, 'wb') as file:
                pickle.dump(distributions, file)
        return distributions,partial_sample

