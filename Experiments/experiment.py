import numpy as np

from Experiments import Tests
# test variables - temp
import cProfile


def your_function_to_profile():
    # params
    show_graph = True
    NN_Powers = np.array([1, 3, 6, 15, 100])
    N_edges_powers = np.array([1, 2, 3, 4, 100])
    Stochastic_powers = np.array([1, 2, 100])

    powers = [NN_Powers, N_edges_powers, Stochastic_powers]
    omitters = ["Bernoulli", "Consecutive Bernoulli"]
    modes = ["Near Neighbors", "N Edges", "Random"]

    half_seen = [np.linspace(0.25, 1, 4)[1], np.linspace(0, 1, 5)[0], 0]
    bernoulli_prob = np.array([0.025 ,0.05, 0.1,0.15,0.2, 0.25, 0.3, 0.35, 0.5, 0.75, 1])
    con_bernoulli_prob = np.array([0, 0.05, 0.1, 0.25, 0.5, 0.75, 1])
    markov_prob = np.linspace(-0.45, 0.45, 10)

    Tests.compare_results_from_different_omitters_transmat_modes_norm_data_size_norm(5, powers, modes, omitters,
                                                                                     half_seen, show=show_graph)
    Tests.compare_pipelines_transmat_mode_with_vs_temp_info_avg_norm_n_readers("Uniform Skips", 15, powers=powers,
                                                                               n_readers=3,
                                                                               transmat_modes=modes,
                                                                               show=show_graph)
    Tests.compare_pipelines_for_different_prob_transmat_mode_vs_temporal_info_norm(5, powers, modes,
                                                                                   show=show_graph)
    Tests.compare_pipelines_for_different_prob_transmat_modes_with_temp_info_vs_prob_norm_data_size_norm_for_pome("Bernoulli", 5,
                                                                                                         transmat_modes=modes,
                                                                                                         probabilities=bernoulli_prob,
                                                                                                         powers=powers,
                                                                                                         show=show_graph)
    # region combined graphs
    Tests.compare_pipelines_for_different_prob_transmat_modes_with_temp_info_vs_prob_norm_data_size_norm(
        "Consecutive Bernoulli", 5,
        transmat_modes=modes,
        probabilities=con_bernoulli_prob,
        powers=powers,
        show=show_graph)


    Tests.compare_pipelines_for_different_prob_transmat_modes_with_temp_info_vs_prob_norm_data_size_norm("Markov Chain",
                                                                                                         5,
                                                                                                         transmat_modes=modes,
                                                                                                         probabilities=markov_prob,
                                                                                                         powers=powers,
                                                                                                         show=show_graph)

    Tests.compare_pipelines_for_different_prob_transmat_modes_with_temp_info_vs_prob_norm_data_size_norm("Geometric",
                                                                                                         15,
                                                                                                         transmat_modes=modes,
                                                                                                         probabilities=bernoulli_prob,
                                                                                                         powers=powers,
                                                                                                         show=show_graph)

    # endregion
    # region special graphs


    # endregion


if __name__ == '__main__':
    cProfile.run('your_function_to_profile()', filename='profile_results.prof')
