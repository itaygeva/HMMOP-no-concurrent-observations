from Data.Readers.brown_corpus_reader import BCReader
import Omission.utils as omitter
from Models.pome_wrapper import pome_wrapper
from Evaluations import utils as evaluations
import numpy as np
import time

# test variables - temp
n_states = 11
n_iter = 20
omission_prob = 0.7


# Record the start time
start_time = time.time()

# import and prepare data
brown_reader = BCReader()
brown_data = brown_reader.get_obs()
brown_emission_prob = brown_reader.get_emission_prob()
omitted_brown_data, omitted_brown_data_idx = omitter.bernoulli_experiments(omission_prob,brown_data)

# fit models and get transition matrices
pome_model = pome_wrapper(n_states, n_iter, brown_emission_prob, freeze_distributions=True)

pome_model.fit(brown_data)
hmmlearn_transmat = pome_model.transmat

"""
hmmlearn_model.fit(omitted_brown_data)
hmmlearn_omitted_transmat = hmmlearn_model.transmat





# run evaluations
evaluations.compare_mat_l1(hmmlearn_omitted_transmat,hmmlearn_transmat)
"""





# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time in seconds
print(f"Program executed in {elapsed_time:.2f} seconds")