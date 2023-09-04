from Data.Readers.brown_corpus_reader import BCReader
import Omission.utils as omitter
from Models.hmmlearn_wrapper import hmmlearn_wrapper
from Evaluations import utils as evaluations

# test variables - temp
n_states = 11
n_iter = 20
omission_prob = 0.7


# import and prepare data
brown_reader = BCReader()
brown_data = brown_reader.get_obs()
brown_emission_prob = brown_reader.get_emission_prob()
omitted_brown_data, omitted_brown_data_idx = omitter.bernoulli_experiments(omission_prob,brown_data)

# fit models and get transition matrices
hmmlearn_model = hmmlearn_wrapper(n_states, n_iter,brown_emission_prob)


hmmlearn_model.fit(omitted_brown_data)
hmmlearn_omitted_transmat = hmmlearn_model.transmat


hmmlearn_model.fit(brown_data)
hmmlearn_transmat = hmmlearn_model.transmat



# run evaluations
evaluations.compare_mat_l1(hmmlearn_omitted_transmat,hmmlearn_transmat)

