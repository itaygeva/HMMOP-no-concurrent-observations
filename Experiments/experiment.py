import numpy as np

from Experiments import Tests
# test variables - temp
import cProfile


def your_function_to_profile():
    show_graph = False
    NN_Powers = np.array([1, 3, 6, 15, 100])
    N_edges_powers = np.array([1, 2, 3, 4, 100])
    Tests.compare_pipelines_for_different_prob_transmat_mode_vs_temporal_info("Markov Chain", 3,
                                                                              transmat_mode="Near Neighbors",
                                                                              probabilities=np.linspace(-0.45, 0.45,
                                                                                                        10),
                                                                              powers=NN_Powers,
                                                                              show=show_graph)
    Tests.compare_pipelines_for_different_prob_transmat_mode_vs_temporal_info("Bernoulli", 3,
                                                                              transmat_mode="Near Neighbors",
                                                                              probabilities=np.linspace(0.25, 1, 4),
                                                                              powers=NN_Powers,
                                                                              show=show_graph)
    Tests.compare_pipelines_for_different_prob_transmat_mode_vs_temporal_info("Consecutive Bernoulli", 3,
                                                                              transmat_mode="Near Neighbors",
                                                                              probabilities=np.linspace(0, 1, 5),
                                                                              powers=NN_Powers,
                                                                              show=show_graph)
    Tests.compare_pipelines_for_different_prob_transmat_mode_vs_temporal_info("Uniform Skips", 3,
                                                                              transmat_mode="Near Neighbors",
                                                                              probabilities=[-0.4, -0.25, 0, 0.25, 0.4],
                                                                              powers=NN_Powers,
                                                                              show=show_graph)

    Tests.compare_pipelines_for_different_prob_transmat_mode_vs_temporal_info("Markov Chain", 3,
                                                                              transmat_mode="N Edges",
                                                                              probabilities=np.linspace(-0.45, 0.45,
                                                                                                        10),
                                                                              powers=N_edges_powers,
                                                                              show=show_graph)
    Tests.compare_pipelines_for_different_prob_transmat_mode_vs_temporal_info("Bernoulli", 3, transmat_mode="N Edges",
                                                                              probabilities=np.linspace(0.25, 1, 4),
                                                                              powers=N_edges_powers,
                                                                              show=show_graph)
    Tests.compare_pipelines_for_different_prob_transmat_mode_vs_temporal_info("Consecutive Bernoulli", 3,
                                                                              transmat_mode="N Edges",
                                                                              probabilities=np.linspace(0, 1, 5),
                                                                              powers=N_edges_powers,
                                                                              show=show_graph)
    Tests.compare_pipelines_for_different_prob_transmat_mode_vs_temporal_info("Uniform Skips", 3,
                                                                              transmat_mode="N Edges",
                                                                              probabilities=[-0.4, -0.25, 0, 0.25, 0.4],
                                                                              powers=N_edges_powers,
                                                                              show=show_graph)


if __name__ == '__main__':
    cProfile.run('your_function_to_profile()', filename='profile_results.prof')
