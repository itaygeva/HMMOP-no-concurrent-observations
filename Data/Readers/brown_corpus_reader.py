from Data.Readers.base_reader import BaseReader
import sys
sys.path.append('../')


class BCReader(BaseReader):
    def __init__(self, path_to_data='raw/ImmuneXpressoResults.csv'):
        super().__init__(path_to_data)

    def read_data(self):
        return

    def get_sentences_length(self):
        return

    def get_transition_matrix(self):
        if not self.is_tagged:
            return False
        return True
