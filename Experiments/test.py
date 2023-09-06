from Data.Readers.brown_corpus_reader import BCReader
import Omission.utils as omitter
from Evaluations import utils as evaluations
import numpy as np
from Models.pome_wrapper import pome_wrapper
import time

# Record the start time
start_time = time.time()


# test variables - temp
n_states = 11
n_iter = 20
omission_prob = 0.7


# import and prepare data
brown_reader = BCReader()
brown_data = brown_reader.get_obs()
brown_tags = brown_reader.get_tags()
brown_emission_prob = brown_reader.get_emission_prob()
omitted_brown_data, omitted_brown_data_idx = omitter.bernoulli_experiments(omission_prob,brown_data)
print("finished preparing")

# fit models and get transition matrices
hmmlearn_model = pome_wrapper(n_states, n_iter, brown_emission_prob)
print("created wrapper")

hmmlearn_model.fit(brown_data)
print("fitted data")
hmmlearn_transmat = hmmlearn_model.transmat
print(hmmlearn_transmat)


emission_prob = hmmlearn_model.emissionprob
evaluations.compare_mat_l1(emission_prob,brown_emission_prob)
evaluations.compare_mat_l1_norm(emission_prob,brown_emission_prob)
"""
hmmlearn_model.fit(omitted_brown_data)
hmmlearn_omitted_transmat = hmmlearn_model.transmat





# run evaluations
evaluations.compare_mat_l1(hmmlearn_omitted_transmat,hmmlearn_transmat)

"""


# Record the end time
end_time = time.time()

# Calculate the duration
duration = end_time - start_time

# Print the duration
print(f"Execution time: {duration} seconds")