from Data.Readers.brown_corpus_reader import BCReader
import Omission.utils as omitter
from hmmlearn import hmm
from Models.hmmlearn_wrapper import hmmlearn_wrapper
from Evaluations import utils as evaluations

# test variables - temp
n_states = 5
n_iter = 20
omission_prob = 0.7


# import and prepare data
brown_data = BCReader.read_data()
omitted_brown_data, omitted_brown_data_idx = omitter.bernoulli_experiments(omission_prob,brown_data)

# fit models and get transition matrices
hmmlearn_model = hmmlearn_wrapper(n_states, n_iter)
hmmlearn_model.fit(omitted_brown_data)
hmmlearn_omitted_transmat = hmmlearn_model.transmat

hmmlearn_model.fit(brown_data)
hmmlearn_transmat = hmmlearn_model.transmat


# run evaluations
evaluations.compare_mat_l1(hmmlearn_omitted_transmat,hmmlearn_transmat)

