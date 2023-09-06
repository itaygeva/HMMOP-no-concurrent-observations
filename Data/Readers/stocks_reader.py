import random

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
import pandas as pd

inspect.getfile(BaseReader)

MAX_LEN = 30
MIN_LEN = 10
class StocksReader(BaseReader):

    def __init__(self, path_to_data='all_stocks_5yr.csv'):
        super().__init__(path_to_data)
        self.n_features = 3
        self.is_tagged = False
        self.n_states = "not inherent in data"
        self._create_stocks_dataset()

    @staticmethod
    def _get_features_from_stocks(stocks_values):
        stocks_open = stocks_values[:, 1]
        stocks_close = stocks_values[:, 4]
        stocks_high = stocks_values[:, 2]
        stocks_low = stocks_values[:, 3]
        frac_change = (stocks_close - stocks_open) / stocks_open
        frac_high = (stocks_high - stocks_open) / stocks_open
        frac_low = (stocks_open - stocks_low) / stocks_open
        features = np.row_stack((frac_change, frac_high, frac_low))
        return np.transpose(np.asarray(features))

    def _create_stocks_dataset(self):
        cache_filename = os.path.join(self.cache_dir, 'stocks_dataset.pkl')
        # Check if the pickle file exists
        if os.path.isfile(cache_filename):
            # If the file exists, load the variable from the file
            with open(cache_filename, 'rb') as file:
                self.dataset = pickle.load(file)
        else:
            # If the file doesn't exist, create the variable and save it to the file
            stocks = pd.read_csv(os.path.join(self.raw_dir, self._path_to_raw))
            stocks_values = stocks.values
            company = 'AAL'
            stock_values = stocks_values[stocks_values[:,6] == company]
            stock_features = self._get_features_from_stocks(stock_values)
            sentences = []
            lengths = []
            start_idx = 0
            end_idx = random.randint(MIN_LEN, MAX_LEN)

            while end_idx <= stock_values.shape[0]:
                sentences.append(stock_features[start_idx:(end_idx-1), :])
                lengths.append(end_idx-start_idx-1)
                start_idx = end_idx
                end_idx = start_idx + random.randint(MIN_LEN, MAX_LEN)

            sentences.append(stock_features[start_idx:-1, :])
            lengths.append(stock_values.shape[0]-start_idx-1)
            self.dataset = {'sentences': sentences, 'tags': None, 'lengths': lengths}
            with open(cache_filename, 'wb') as file:
                pickle.dump(self.dataset, file)

if __name__ == '__main__':
    s = StocksReader()
    a = s.get_obs()
    b = s.get_lengths()
    print(a)