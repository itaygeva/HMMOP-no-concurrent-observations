class BaseReader:
    is_tagged = False

    def __init__(self, path_to_data):
        self._path_to_raw = path_to_data

    def read_data(self):
        dic= self.dataset.copy()
        del(dic['lengths'])
        return dic

    def get_sentences_length(self):
        return self.dataset['lengths']

    def get_observation_dim(self):
        return self.dim

    def get_n_state(self):
        return self.n_state

    def get_transition_matrix(self):
        if not self.is_tagged:
            return False
        return True
