from Experiments import Tests
# test variables - temp
import cProfile


def your_function_to_profile():
    show_graph = True
    Tests.compare_pipelines_vs_iter_pass_all(10, 5, 20, show=show_graph)
    Tests.compare_pipelines_for_different_prob("Bernoulli", 5, probabilities=[0.05, 0.1, 0.325, 0.55, 0.775, 1],
                                               show=show_graph)
    Tests.compare_pipelines_for_different_prob("Geometric", 5, probabilities=[0.1, 0.325, 0.55, 0.775, 1],
                                               show=show_graph)
    Tests.compare_pipelines_for_different_prob("Consecutive Bernoulli", 5,
                                               probabilities=[0, 0.05, 0.1, 0.325, 0.55, 0.775, 1], show=show_graph)
    Tests.compare_pipelines_for_different_prob("Markov Chain", 5,
                                               probabilities=[-0.4, -0.25, 0, 0.25, 0.4], show=show_graph)
    Tests.compare_pipelines_vs_iter_pass_all_different_sample_len(5, 20, sentence_length_arr=[20, 10, 5, 100],
                                                                  n_samples_arr=[1000, 2000, 4000, 200],
                                                                  show=show_graph)

    """    Tests.compare_pipelines_for_different_prob_vs_iter("Bernoulli", 5,n_iter=20, probabilities=[0.05, 0.1, 0.325, 0.55, 0.775, 1],
                                               show=show_graph)
    Tests.compare_pipelines_for_different_prob_vs_iter("Geometric", 5, n_iter=20,probabilities=[0.05, 0.1, 0.325, 0.55, 0.775, 1],
                                               show=show_graph)
    Tests.compare_pipelines_for_different_prob_vs_iter("Consecutive Bernoulli", 5, n_iter=20,
                                               probabilities=[0, 0.05, 0.1, 0.325, 0.55, 0.775, 1], show=show_graph)
    Tests.compare_pipelines_for_different_prob_vs_iter("Markov Chain", 5, n_iter=20,
                                               probabilities=[-0.5, -0.25, 0, 0.25, 0.5], show=show_graph)"""

if __name__ == '__main__':
    cProfile.run('your_function_to_profile()', filename='profile_results.prof')
