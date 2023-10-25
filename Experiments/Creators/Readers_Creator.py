from Data.Readers.stocks_reader import StocksReader
from Data.Readers.synthetic_reader import synthetic_reader
from Data.Readers.brown_corpus_reader import BCReader
from Experiments.Creators.Base_Creator import Base_Creator
from Experiments.Creators.utils import *


class Readers_Creator(Base_Creator):

    def __init__(self, config_path=None):
        """
        :param config_path: the path to the JSON configuration file
        """
        super().__init__(config_path, "readers")
        self.class_to_builder_dict = {
            "brown_corpus_reader": self._build_brown_corpus_reader,
            "synthetic_reader": self._build_synthetic_reader,
            "stocks_reader": self._build_stocks_reader
        }

    def _build_brown_corpus_reader(self, reader_config):
        """
        creates a brown corpus reader
        :param reader_config: the reader configuration loaded from the JSON file
        :return: a brown corpus reader
        """
        return BCReader()

    def _build_synthetic_reader(self, reader_config):
        """
        creates a synthetic reader
        :param reader_config: the reader configuration loaded from the JSON file
        :return: a synthetic reader
        """
        n_states = get_value_or_default("Number of States for Generation", reader_config, self.default)
        n_samples = get_value_or_default("Number of Samples", reader_config, self.default)
        return synthetic_reader(n_states, n_samples, 1)

    def _build_stocks_reader(self, reader_config):
        """
        creates a stocks reader
        :param reader_config: the reader configuration loaded from the JSON file
        :return: a stocks reader
        """
        return StocksReader()
