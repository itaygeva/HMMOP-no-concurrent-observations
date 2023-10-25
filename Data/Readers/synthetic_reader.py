import torch

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
from pomegranate.hmm import DenseHMM
from Config.Config import synthetic_reader_config

inspect.getfile(BaseReader)


class synthetic_reader(BaseReader):

    def __init__(self, config: synthetic_reader_config, **kwargs):
        super().__init__(config, **kwargs)
        self.n_features = 1
        self.is_tagged = True
        self.tag_dict = {}
        self.word_dict = {}
        self.transition_mat = torch.zeros([self._config.n_components, self._config.n_components])
        self.end_probs = torch.zeros([self._config.n_components, 1])
        self._init_hmm_model(*self._config.args)
        self._generate_samples(self._config.n_samples)
    def _generate_samples(self, n_samples):
        self.dataset['sentences'] = self.model.sample(n_samples)
        self.dataset['lengths'] = [sample.shape[0] for sample in self.dataset['sentences']]
        self.delete_one_word_sentences()  # this is so we won't get an error in the gibbs sampler

    def _init_hmm_model(self, *args):
        """
        this function initialize the model for the synthetic sample. this functions has two modes:
            _init_hmm_model(self, model_num): used for prying a model that is already in cache
            _init_hmm_model(self, min_mu_gap, sigma): used for creating a new model.
        """
        cache_filename = os.path.join(self.cache_dir, 'synthetic_model_' + str(args[0]) + '.pkl')
        # Check if the pickle file exists
        if os.path.isfile(cache_filename) and len(args) == 1:
            # If the file exists, load the variable from the file
            with open(cache_filename, 'rb') as file:
                self.model = pickle.load(file)
        else:
            if len(args) == 1:
                min_mu_gap = 10
                sigma = 1
            else:
                min_mu_gap, sigma = args
            N = self._config.n_components * 3
            mues = list(range(N))
            mues = [mue * min_mu_gap for mue in mues]
            total_distributions = np.array(
                [pomegranate.distributions.Normal([float(mues[i])], [float(sigma)], 'diag') for i in range(N)])
            self.distributions = total_distributions[np.random.choice(range(N), self._config.n_components, replace=False)]

            for s in range(self._config.n_components):
                temp = np.random.choice(range(1000), self._config.n_components + 1, replace=False)
                self.transition_mat[s, :] = torch.from_numpy(temp[:-1] / np.sum(temp))
                self.end_probs[s] = temp[-1] / np.sum(temp)

            temp = np.random.choice(range(1000), self._config.n_components, replace=False)
            self.start_probs = torch.from_numpy(temp / np.sum(temp))

            # temp = np.random.choice(range(1000), self.n_states, replace=False)
            # self.end_probs = torch.from_numpy(temp / np.sum(temp))
            self.end_probs = self.end_probs.squeeze()

            self.model = DenseHMM(self.distributions, self.transition_mat, self.start_probs, self.end_probs)

            i = 0
            flag = True
            while flag:
                cache_filename = os.path.join(self.cache_dir, 'synthetic_model_' + str(i) + '.pkl')
                if not os.path.isfile(cache_filename):
                    flag = False
                    with open(cache_filename, 'wb') as file:
                        pickle.dump(self.model, file)
                i += 1

    def delete_one_word_sentences(self):
        no_one_word_data = [(length, sentences) for (length, sentences) in
                            zip(self.dataset['lengths'], self.dataset['sentences']) if length > 3]
        self.dataset['lengths'], self.dataset['sentences'] = zip(*no_one_word_data)


if __name__ == '__main__':
    reader = synthetic_reader(10, 5, 1)
    b = reader.get_obs()
    print(b)
