import os
import random

import numpy as np
import pandas as pd

from Config.Config import stocks_reader_config
from Data.Readers.base_reader import base_reader


class stocks_reader(base_reader):

    def __init__(self, config: stocks_reader_config):
        super().__init__(config)
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
        # If the file doesn't exist, create the variable and save it to the file
        stocks = pd.read_csv(os.path.join(self._config.raw_dir, self._config.path_to_data))
        stocks_values = stocks.values
        stock_values = stocks_values[stocks_values[:, 6] == self._config.company]
        stock_features = self._get_features_from_stocks(stock_values)
        sentences = []
        lengths = []
        start_idx = 0
        end_idx = random.randint(self._config.min_length, self._config.max_length)

        ## TODO: I think there might be a better way, of generating all of the start and end indx
        ## and then splitting it all together. Maybe numpy has, or itertools. MAYBE EVEN USING convert_to_our_format
        while end_idx <= stock_values.shape[0]:
            sentences.append(stock_features[start_idx:(end_idx - 1), :])
            lengths.append(end_idx - start_idx - 1)
            start_idx = end_idx
            end_idx = start_idx + random.randint(self._config.min_length, self._config.max_length)

        sentences.append(stock_features[start_idx:-1, :])
        lengths.append(stock_values.shape[0] - start_idx - 1)
        self.dataset = {'sentences': sentences, 'tags': None, 'lengths': lengths}


