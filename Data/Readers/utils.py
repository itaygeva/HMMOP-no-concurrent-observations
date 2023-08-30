import os
from itertools import chain
from collections import namedtuple, OrderedDict
import snowballstemmer
import string

Sentence = namedtuple("Sentence", "words tags")


class Utils:

    @staticmethod
    def corpus_read_data(filename):
        """Read tagged sentence data"""
        with open(filename, 'r') as f:
            sentence_lines = [l.split("\n") for l in f.read().split("\n\n")]
        return OrderedDict(((s[0], Sentence(*zip(*[l.strip().split("\t")
                                                   for l in s[1:]]))) for s in sentence_lines if s[0]))

    @staticmethod
    def corpus_read_tags(filename):
        """Read a list of word tag classes"""
        with open(filename, 'r') as f:
            tags = f.read().split("\n")
        return frozenset(tags)

    """ @staticmethod
    def corpus_dataset(path_to_raw):
        tagset = Utils.corpus_read_tags(path_to_raw(1))
        sentences = Utils.corpus_read_data(path_to_raw(0))
        keys = tuple(sentences.keys())

        stemmer = snowballstemmer.stemmer('english')
        num_to_symbol = lambda x: x if not x.isnumeric() else "#"
        non_relevant_words = string.punctuation + '``' + '.' + '--' + "''" + ','

        words_sequences = []
        tag_sequences = []

        for idx, sentence in sentences:
            clean_tuples = [(num_to_symbol(stemmer.stemWord(word)), tag) for word, tag in zip(sentence.words,
                                                                                              sentence.tags) if
                            word not in non_relevant_words]
            if len(clean_tuples) < 2: continue
            words_sequences.append([word for word, tag in clean_tuples])
            tag_sequences.append([tag for word, tag in clean_tuples])

        length_sequences = sum(1 for _ in chain(*(s.words for s in sentences.values())))

        return {'words': words_sequences, 'tags': tag_sequences, 'lengths': length_sequences}"""
