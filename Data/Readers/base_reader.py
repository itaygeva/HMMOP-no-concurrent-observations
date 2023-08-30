import os
import inspect
import numpy as np
class BaseReader:
    # This class is the parent of all reader and handles the basic query methods they all share
    is_tagged = False
    n_features = 1
    n_states = 1
    dataset={'sentences':[],'tags':[],'lengths':[]}

    def __init__(self, path_to_data):
        self._path_to_raw = path_to_data
        data_dir=os.path.dirname(os.path.dirname(inspect.getfile(BaseReader)))
        self.raw_dir=os.path.join(data_dir,'Raw')

    def get_obs(self):
        # returns the observations as a list of sentences. each sentence is a np.array of size (n_obs,n_features
        return self.dataset['sentences']

    def get_tags(self):
        # returns the tags as a list of tag sentences. each tag sentence is a np.array of size(n_obs)
        return self.dataset['tags']

    def get_lengths(self):
        # returns a list of the sentence lengths.
        return self.dataset['lengths']

    def get_n_features(self):
        # returns the number of features per observation.
        return self.n_features

    def get_n_states(self):
        # returns the number of states in the HMM model.
        return self.n_states

    def get_if_tagged(self):
        # returns a boolean whether  the dataset is tagged or not.
        return self.is_tagged

    def convert_to_our_format(self):
        # works on the datasets as one list of all words, with appropriate lengths and converts to the desired format(see above)
        idx = 0
        total_length = 0
        sentences = []
        tags = []
        for length in self.dataset['lengths']:
            total_length += length
            sentence = []
            s_tags = []
            while idx < total_length:
                idx += 1
                sentence.append(self.dataset['words'][idx,:])
                if self.is_tagged:
                    s_tags.append(self.dataset['tags'][idx])
            np_sentence=np.array(sentence)
            np_s_tags = np.array(s_tags)
        sentences.append(np_sentence)
        tags.append(np_s_tags)
        self.dataset['words'] = sentences
        if self.is_tagged:
            self.dataset['tags'] = tags
