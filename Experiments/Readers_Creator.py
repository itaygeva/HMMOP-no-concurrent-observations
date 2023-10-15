from Data.Readers.stocks_reader import StocksReader
from Data.Readers.synthetic_reader import SyntheticReader
from Data.Readers.brown_corpus_reader import BCReader
from Experiments.Base_Creator import Base_Creator
from utils import *
import json


class Readers_Creator(Base_Creator):

    def __init__(self, config_path=None):
        super().__init__(config_path)
        self.instances_type = "readers"
        self.class_to_builder_dict = {
            "brown_corpus_reader": self._build_brown_corpus_reader,
            "synthetic_reader": self._build_synthetic_reader,
            "stocks_reader": self._build_stocks_reader
        }

    def _build_brown_corpus_reader(self, reader_config):
        return BCReader()

    def _build_synthetic_reader(self, reader_config):
        n_states = get_value_or_default("Number of States for Generation", reader_config, self.default)
        n_samples = get_value_or_default("Number of Samples", reader_config, self.default)
        return SyntheticReader(n_states, n_samples, 1)

    def _build_stocks_reader(self, reader_config):
        return StocksReader()
