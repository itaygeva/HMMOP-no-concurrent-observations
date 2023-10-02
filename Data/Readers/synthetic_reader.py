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

inspect.getfile(BaseReader)


class SyntheticReader(BaseReader):

    def __init__(self, n_states, *args):
        super().__init__('')
        self.n_features = 1
        self.is_tagged = True
        self.tag_dict = {}
        self.word_dict = {}
        self.n_states = n_states
        self._init_hmm_model(*args)
        self._generate_samples(10000)
        print('bla')

    def _generate_samples(self, n_samples):
        self.dataset['sentences'] = self.model.sample(n_samples)
        print('hi')
        self.dataset['lengths'] = [sample.shape[0] for sample in self.dataset['sentences']]

    def _init_hmm_model(self, *args):
        """
        this function initialize the model for the synthetic sample. this functions has two modes:
            _init_hmm_model(self, model_num): used for prying a model that is already in cache
            _init_hmm_model(self, min_mu_gap, sigma): used for creating a new model.
        """
        cache_filename = os.path.join(self.cache_dir, 'synthetic_model_'+str(args[0])+'.pkl')
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
            N = self.n_states * 3
            mues = list(range(N))
            mues = [mue*min_mu_gap for mue in mues]
            total_distributions = np.array([pomegranate.distributions.Normal([float(mues[i])], [float(sigma)],'diag') for i in range(N)])
            self.distributions = total_distributions[np.random.choice(range(N), self.n_states, replace=False)]

            self.transition_mat = torch.zeros([self.n_states, self.n_states])
            self.end_probs = torch.zeros([self.n_states, 1])
            for s in range(self.n_states):
                temp = np.random.choice(range(1000), self.n_states+1, replace=False)
                self.transition_mat[s,:] = torch.from_numpy(temp[:-1]/np.sum(temp))
                self.end_probs[s] = temp[-1]/np.sum(temp)

            temp = np.random.choice(range(1000), self.n_states, replace=False)
            self.start_probs = torch.from_numpy(temp / np.sum(temp))

            # temp = np.random.choice(range(1000), self.n_states, replace=False)
            # self.end_probs = torch.from_numpy(temp / np.sum(temp))
            self.end_probs = self.end_probs.squeeze()

            self.model = DenseHMM(self.distributions, self.transition_mat, self.start_probs, self.end_probs)

            i = 0
            flag = True
            while flag:
                cache_filename = os.path.join(self.cache_dir, 'synthetic_model_'+str(i)+'.pkl')
                if not os.path.isfile(cache_filename):
                    flag = False
                    with open(cache_filename, 'wb') as file:
                        pickle.dump(self.model, file)
                i += 1

if __name__ == '__main__':
    reader = SyntheticReader(10, 5, 1)
    b = reader.get_obs()
    print(b)
    