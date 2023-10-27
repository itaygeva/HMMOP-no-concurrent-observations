import os
import string
from collections import namedtuple, OrderedDict

import numpy as np
import snowballstemmer

from Config.Config import brown_corpus_reader_config
from Data.Readers.base_reader import base_reader


class brown_corpus_reader(base_reader):

    def __init__(self, config: brown_corpus_reader_config):
        super().__init__(config)
        self.sentence = namedtuple("Sentence", "words tags")
        self.tag_dict = {}
        self.word_dict = {}
        self.tag_appearances = np.zeros(
            (self._config.n_components, 1))  # list of number of time the tag exist per tag idx
        self._corpus_dataset()
        self._calculate_emission_prob()

    def _corpus_read_data(self, filename):
        """Read tagged sentence data"""
        with open(os.path.join(self._config.raw_dir, filename), 'r') as f:
            sentence_lines = [l.split("\n") for l in f.read().split("\n\n")]
        return OrderedDict(((s[0], self.sentence(*zip(*[l.strip().split("\t")
                                                   for l in s[1:]]))) for s in sentence_lines if s[0]))

    def _corpus_read_tags(self, filename):
        """Read a list of word tag classes"""
        with open(os.path.join(self._config.raw_dir, filename), 'r') as f:
            tags = f.read().split("\n")
        return frozenset(tags)

    def _corpus_dataset(self):
        ## TODO: Too complicated for me. Just made sure the format is fine. Need to go over with Geva
        tagset = self._corpus_read_tags(self._config.path_to_tags)
        sentences = self._corpus_read_data(self._config.path_to_data)
        keys = tuple(sentences.keys())

        stemmer = snowballstemmer.stemmer('english')
        num_to_symbol = lambda x: x if not x.isnumeric() else "#"
        non_relevant_words = string.punctuation + '``' + '.' + '--' + "''" + ','

        words_sequences = []
        tag_sequences = []

        word_idx = 0
        tag_idx = 0
        for idx, sentence in sentences.items():
            clean_tuples = [(num_to_symbol(stemmer.stemWord(word)), tag) for word, tag in zip(sentence.words,
                                                                                              sentence.tags) if
                            word not in non_relevant_words]
            if len(clean_tuples) < 2: continue

            for word, tag in clean_tuples:
                if word not in self.word_dict:
                    self.word_dict[word] = word_idx
                    word_idx += 1
                if tag not in self.tag_dict:
                    self.tag_dict[tag] = tag_idx
                    tag_idx += 1
                self.tag_appearances[self.tag_dict[tag]] += 1

            words_sequences.append(np.array([self.word_dict[word] for word, tag in clean_tuples]))
            tag_sequences.append(np.array([self.tag_dict[tag] for word, tag in clean_tuples]))

        # length_sequences = sum(1 for _ in chain(*(s.words for s in sentences.values())))
        lengths = np.array([len(sequence) for sequence in tag_sequences])
        self.dataset = {'sentences': words_sequences, 'tags': tag_sequences, 'lengths': lengths}

    def _calculate_emission_prob(self):
        # If the file doesn't exist, create the variable and save it to the file
        self.emission_prob = np.zeros((self.n_states, len(self.word_dict)))
        for s_idx, sentence in enumerate(self.dataset['sentences']):
            tag_seq = self.dataset['tags'][s_idx]
            for o_idx, obs in enumerate(sentence):
                tag = tag_seq[o_idx]
                self.emission_prob[tag][obs] += 1 / self.tag_appearances[tag]
