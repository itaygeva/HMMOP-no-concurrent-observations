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

Sentence = namedtuple("Sentence", "words tags")
inspect.getfile(BaseReader)


class BCReader(BaseReader):


    def __init__(self, path_to_data=('brown-universal.txt', 'tags-universal.txt')):
        super().__init__(path_to_data)
        self.n_features = 1
        self.is_tagged = True
        self.tag_dict = {}
        self.word_dict = {}
        self.n_states = 11
        self.tag_appearances = np.zeros((self.n_states , 1))  # list of number of time the tag exist per tag idx
        self._corpus_dataset()
        self._calculate_emission_prob()


    def _corpus_read_data(self, filename):
        """Read tagged sentence data"""
        with open(os.path.join(self.raw_dir, filename), 'r') as f:
            sentence_lines = [l.split("\n") for l in f.read().split("\n\n")]
        return OrderedDict(((s[0], Sentence(*zip(*[l.strip().split("\t")
                                                   for l in s[1:]]))) for s in sentence_lines if s[0]))

    def _corpus_read_tags(self, filename):
        """Read a list of word tag classes"""
        with open(os.path.join(self.raw_dir, filename), 'r') as f:
            tags = f.read().split("\n")
        return frozenset(tags)

    def _corpus_dataset(self):
        tagset = self._corpus_read_tags(self._path_to_raw[1])
        sentences = self._corpus_read_data(self._path_to_raw[0])
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
        lengths= np.array([len(sequence) for sequence in tag_sequences])
        self.dataset = {'sentences': words_sequences, 'tags': tag_sequences, 'lengths': lengths}

    def _calculate_emission_prob(self):
        self.emission_prob=np.zeros((self.n_states,len(self.word_dict)))
        for s_idx, sentence in enumerate(self.dataset['sentences']):
            tag_seq = self.dataset['tags'][s_idx]
            for o_idx , obs in enumerate(sentence):
                tag = tag_seq[o_idx]
                self.emission_prob[tag][obs] += 1 / self.tag_appearances[tag]




