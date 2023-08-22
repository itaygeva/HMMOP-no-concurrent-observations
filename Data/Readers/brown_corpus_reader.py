from Data.Readers.base_reader import BaseReader
import sys
from Data.Readers.utils import Utils
from itertools import chain

sys.path.append('../')


class BCReader(BaseReader):
    def __init__(self, path_to_data=('Raw/brown-universal.txt', 'Raw/tags-universal.txt')):
        super().__init__(path_to_data)
        self.dim = 1
        self.is_tagged = True
        self.n_state = 12
        self.dataset = Utils.corpus_dataset(self._path_to_raw)

    def get_transition_matrix(self):
        if not self.is_tagged:
            return False
        return True
