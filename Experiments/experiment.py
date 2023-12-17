from Evaluations import Tests
# test variables - temp
import time
import cProfile

def your_function_to_profile():
    #print("Testing with synthetic reader:")
    #Tests.compare_synthetic_transmat()
    #print("Testing with hmm synthetic reader:")
    Tests.compare_pipelines_vs_iter()
    Tests.compare_pipelines_for_different_sigmas()
    #Tests.compare_pipelines_for_different_bernoulli_prob()

if __name__ == '__main__':
    cProfile.run('your_function_to_profile()', filename='profile_results.prof')



    """
    n_states = 3
    n_iter = 1000
    omission_prob = 1
    n_samples = 100000

    print("generating data")
    reader = SyntheticReader(n_states, n_samples, 6)
    data = reader.get_obs()
    data = [np.squeeze(sentence.numpy()) for sentence in data]

    og_transmat = reader.transition_mat

    print("fitting")
    model = hmmlearn_wrapper(n_states, n_iter,"Gaussian")
    model.fit(data)

    found_transmat = torch.from_numpy(model.transmat)
    print(og_transmat)
    print(found_transmat)
    print(torch.sum(found_transmat))
    evaluations.compare_mat_l1_norm(og_transmat, found_transmat)
"""
"""
    gibbs_sampler = gibbs_sampler_wrapper(n_components=n_states, n_iter=n_iter)
    gibbs_sampler.fit(data)
    found_transmat = gibbs_sampler.transmat
    print(og_transmat)
    print(found_transmat)
    evaluations.compare_mat_l1_norm(og_transmat, found_transmat)
"""

"""
# Record the start time
start_time = time.time()

# import and prepare data
brown_reader = BCReader()
brown_data = brown_reader.get_obs()
brown_emission_prob = brown_reader.get_emission_prob()
omitted_brown_data, omitted_brown_data_idx = omitter.bernoulli_experiments(omission_prob,brown_data)


# fit models and get transition matrices
print("creating model")
pome_model = pome_wrapper(n_states, n_iter, brown_emission_prob, freeze_distributions=True)

print("fitting full data")
pome_model.fit(brown_data)


pome_transmat = pome_model.transmat
print(pome_transmat)

print("creating model")
pome_model_omitted = pome_wrapper(n_states, n_iter, brown_emission_prob, freeze_distributions=True)
print("fitting omitted data")
pome_model_omitted.fit(omitted_brown_data,omitted_brown_data_idx)
pome_transmat_omitted = pome_model_omitted.transmat



print(pome_transmat_omitted)
evaluations.compare_mat_l1(pome_transmat.numpy(), pome_transmat_omitted.numpy())
evaluations.compare_mat_l1_norm(pome_transmat.numpy(), pome_transmat_omitted.numpy())


# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time in seconds
print(f"Program executed in {elapsed_time:.2f} seconds")
"""
