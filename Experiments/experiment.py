from Evaluations import Tests
# test variables - temp
import time
import cProfile


def your_function_to_profile():
    # print("Testing with synthetic reader:")
    # Tests.compare_synthetic_transmat()
    # print("Testing with hmm synthetic reader:")
    Tests.compare_pipelines_vs_iter(10, 5, 20)
    # Tests.compare_pipelines_for_different_sigmas()
    # Tests.compare_pipelines_for_different_bernoulli_prob()


if __name__ == '__main__':
    cProfile.run('your_function_to_profile()', filename='profile_results.prof')