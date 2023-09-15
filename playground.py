import numpy as np
import pomegranate.hmm as hmm
import pomegranate.distributions as distributions
from Data.Readers.brown_corpus_reader import BCReader
import Omission.utils as omitter
from Models.hmmlearn_wrapper import hmmlearn_wrapper
from Evaluations import utils as evaluations
arr = np.array([1,2,3])
print(arr.reshape(-1,1).shape)

