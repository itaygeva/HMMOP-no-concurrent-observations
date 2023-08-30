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
        self.dim = 1
        self.is_tagged = True
        self.n_state = 12
        self.corpus_dataset()

    def corpus_read_data(self, filename):
        """Read tagged sentence data"""
        with open(os.path.join(self.raw_dir, filename), 'r') as f:
            sentence_lines = [l.split("\n") for l in f.read().split("\n\n")]
        return OrderedDict(((s[0], Sentence(*zip(*[l.strip().split("\t")
                                                   for l in s[1:]]))) for s in sentence_lines if s[0]))

    def corpus_read_tags(self, filename):
        """Read a list of word tag classes"""
        with open(os.path.join(self.raw_dir, filename), 'r') as f:
            tags = f.read().split("\n")
        return frozenset(tags)

    def corpus_dataset(self):
        tagset = self.corpus_read_tags(self._path_to_raw[1])
        sentences = self.corpus_read_data(self._path_to_raw[0])
        keys = tuple(sentences.keys())

        stemmer = snowballstemmer.stemmer('english')
        num_to_symbol = lambda x: x if not x.isnumeric() else "#"
        non_relevant_words = string.punctuation + '``' + '.' + '--' + "''" + ','

        words_sequences = []
        tag_sequences = []

        for idx, sentence in sentences.items():
            clean_tuples = [(num_to_symbol(stemmer.stemWord(word)), tag) for word, tag in zip(sentence.words,
                                                                                              sentence.tags) if
                            word not in non_relevant_words]
            if len(clean_tuples) < 2: continue
            tag_dict = {}
            word_dict = {}
            word_idx = 0
            tag_idx = 0
            for word, tag in clean_tuples:
                if word not in word_dict:
                    word_dict[word] = word_idx
                    word_idx += 1
                if tag not in tag_dict:
                    tag_dict[tag] = tag_idx
                    tag_idx += 1

            words_sequences.append(np.array([word_dict[word] for word, tag in clean_tuples]))
            tag_sequences.append(np.array([tag_dict[tag] for word, tag in clean_tuples]))

        length_sequences = sum(1 for _ in chain(*(s.words for s in sentences.values())))
        self.dataset = {'sentences': words_sequences, 'tags': tag_sequences, 'lengths': length_sequences}

