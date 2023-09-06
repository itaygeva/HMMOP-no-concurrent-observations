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


class StocksReader(BaseReader):

    def __init__(self, path_to_data='all_stocks_5yr.csv'):
        super().__init__(path_to_data)
        self.n_features = 3
        self.is_tagged = False
        self.n_states = "not inherent in data"

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
        return np.asarray(features)

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
            sentences=[]
            start_idx = 0
            end_idx = 0





            self.dataset = {'sentences': sentences, 'tags': None, 'lengths': lengths}
            with open(cache_filename, 'wb') as file:
                pickle.dump(self.dataset, file)

