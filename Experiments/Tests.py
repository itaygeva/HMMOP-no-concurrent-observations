import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

import Config.Config
from Experiments.utils import create_config_dataclass_objects as create_config
from Experiments.utils import create_and_fit_pipeline as create_pipeline
from Experiments.utils import load_or_initialize_pipeline as create_pipeline_from_configs
from Pipelines.pipeline import pipeline
from Experiments.evalUtils import *
from Config.Config import *

height = 10
width = 7


def simple_valid_pipeline(pipeline_to_check: pipeline):
    print(pipeline_to_check.transmat)
    print(pipeline_to_check.startprob)
    print(len(pipeline_to_check.transmat_list))
    print(len(pipeline_to_check.startprob_list))


def simple_validate_pipelines():
    ## TODO: Create default for readers, omitters, models
    ## TODO: Create dataclasses for configurations, as seen in playground.
    # Create dict of attributes from the dataclass, and fill it up using json and default.
    # Then create the dataclass and pass to the object
    # and then the model etc will hold a config obj

    # region pome, pass all, simple valid
    """pipeline_pome_stocks: pipeline = create_pipeline("Stocks", "Pass All", "Pomegranate - Stocks")
    print(f"finished creating {pipeline_pome_stocks.__str__()}")
    pipeline_pome_syn: pipeline = create_pipeline("Synthetic", "Pass All", "Pomegranate - Synthetic")
    print(f"finished creating {pipeline_pome_syn.__str__()}")
    pipeline_pome_brown: pipeline = create_pipeline("Brown Corpus", "Pass All", "Pomegranate - Brown")
    print(f"finished creating {pipeline_pome_brown.__str__()}")

    simple_valid_pipeline(pipeline_pome_stocks)
    simple_valid_pipeline(pipeline_pome_syn)
    simple_valid_pipeline(pipeline_pome_brown)"""
    # endregion

    # region pome, bernoulli, simple valid

    """pipeline_pome_stocks: pipeline = create_pipeline("Stocks", "Bernoulli", "Pomegranate - Stocks")
    print(f"finished creating {pipeline_pome_stocks.__str__()}")
    pipeline_pome_syn: pipeline = create_pipeline("Synthetic", "Bernoulli", "Pomegranate - Synthetic")
    print(f"finished creating {pipeline_pome_syn.__str__()}")
    pipeline_pome_brown: pipeline = create_pipeline("Brown Corpus", "Bernoulli", "Pomegranate - Brown")
    print(f"finished creating {pipeline_pome_brown.__str__()}")

    simple_valid_pipeline(pipeline_pome_stocks)
    simple_valid_pipeline(pipeline_pome_syn)
    simple_valid_pipeline(pipeline_pome_brown)"""
    # endregion

    # region gibbs, pass all, simple valid
    pipeline_gibbs_syn: pipeline = create_pipeline("Synthetic", "Pass All", "Gibbs Sampler")
    print(f"finished creating {pipeline_gibbs_syn.__str__()}")

    simple_valid_pipeline(pipeline_gibbs_syn)
    print(pipeline_gibbs_syn.startprob_list)

    # endregion
    # region gibbs, bernoulli, simple valid
    pipeline_gibbs_syn: pipeline = create_pipeline("Synthetic", "Bernoulli", "Gibbs Sampler")
    print(f"finished creating {pipeline_gibbs_syn.__str__()}")

    simple_valid_pipeline(pipeline_gibbs_syn)
    print(pipeline_gibbs_syn.startprob_list)

    # endregion


def compare_synthetic_transmat():
    pipeline_pome_syn: pipeline = create_pipeline("Synthetic", "Pass All", "Pomegranate - Synthetic")
    print(f"finished creating {pipeline_pome_syn}")
    pipeline_hmm_syn: pipeline = create_pipeline("Synthetic", "Pass All", "Hmmlearn")
    print(f"finished creating {pipeline_hmm_syn}")
    pipeline_gibbs_syn: pipeline = create_pipeline("Synthetic", "Pass All", "Gibbs Sampler")
    print(f"finished creating {pipeline_gibbs_syn}")
    pipeline_gt_syn: pipeline = create_pipeline("Synthetic", "Pass All", "Ground Truth")
    print(f"finished creating {pipeline_gt_syn}")

    print(pipeline_pome_syn.reader.transmat)
    print(pipeline_hmm_syn.reader.transmat)
    print(pipeline_gibbs_syn.reader.transmat)
    print(pipeline_gt_syn.reader.transmat)

    print("matrices")
    print(pipeline_pome_syn.transmat)
    print(pipeline_hmm_syn.transmat)
    print(pipeline_gibbs_syn.transmat)
    print(pipeline_gt_syn.transmat)

    print("matrices norm")
    print(np.sum(pipeline_pome_syn.transmat, axis=1))
    print(np.sum(pipeline_hmm_syn.transmat, axis=1))
    print(np.sum(pipeline_gibbs_syn.transmat, axis=1))
    print(np.sum(pipeline_gt_syn.transmat, axis=1))

    print("matrices compared l1 norm")
    compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_pome_syn.transmat)
    compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_hmm_syn.transmat)
    compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_gibbs_syn.transmat)

    print("means")
    print(pipeline_pome_syn.means)
    print(pipeline_hmm_syn.means)
    print(pipeline_gt_syn.means)


def compare_hmm_synthetic_transmat():
    pipeline_pome_syn: pipeline = create_pipeline("HMM - Synthetic", "Pass All", "Pomegranate - Synthetic")
    print(f"finished creating {pipeline_pome_syn}")
    pipeline_hmm_syn: pipeline = create_pipeline("HMM - Synthetic", "Pass All", "Hmmlearn")
    print(f"finished creating {pipeline_hmm_syn}")
    pipeline_gibbs_syn: pipeline = create_pipeline("HMM - Synthetic", "Pass All", "Gibbs Sampler")
    print(f"finished creating {pipeline_gibbs_syn}")
    pipeline_gt_syn: pipeline = create_pipeline("HMM - Synthetic", "Pass All", "Ground Truth")
    print(f"finished creating {pipeline_gt_syn}")

    print("matrices")
    print(pipeline_pome_syn.transmat)
    print(pipeline_hmm_syn.transmat)
    print(pipeline_gibbs_syn.transmat)
    print(pipeline_gt_syn.transmat)
    print()

    print("matrices norm")
    print(np.sum(pipeline_pome_syn.transmat, axis=1))
    print(np.sum(pipeline_hmm_syn.transmat, axis=1))
    print(np.sum(pipeline_gibbs_syn.transmat, axis=1))
    print(np.sum(pipeline_gt_syn.transmat, axis=1))

    print("matrices compared l1 norm")
    compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_pome_syn.transmat)
    compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_hmm_syn.transmat)
    compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_gibbs_syn.transmat)

    print("means")
    print(pipeline_pome_syn.means)
    print(pipeline_hmm_syn.means)
    print(pipeline_gt_syn.means)


def compare_pipelines_for_different_sigmas():
    readers = ["Synthetic Hard"]
    num_of_run = 5
    results = np.empty((2, 1, num_of_run))
    fig, ax = plt.subplots(num_of_run, 1, sharex=True, sharey=True, figsize=(10, 6))
    for j in range(1, num_of_run + 1):
        for i, reader in enumerate(readers):
            pipeline_pome_syn: pipeline = create_pipeline(reader, "Pass All", "Pomegranate - Synthetic" + str(j))
            print(f"finished creating {pipeline_pome_syn}")
            """pipeline_hmm_syn: pipeline = create_pipeline(reader, "Pass All", "Hmmlearn" + str(j))
            print(f"finished creating {pipeline_hmm_syn}")"""
            pipeline_gibbs_syn: pipeline = create_pipeline(reader, "Pass All", "Gibbs Sampler" + str(j))
            print(f"finished creating {pipeline_gibbs_syn}")
            pipeline_gt_syn: pipeline = create_pipeline(reader, "Pass All", "Ground Truth" + str(j))
            print(f"finished creating {pipeline_gt_syn}")

            print("matrices compared l1 norm")
            pome_perm = find_optimal_permutation(pipeline_pome_syn.means, pipeline_gt_syn.means)
            # hmm_perm = find_optimal_permutation(pipeline_hmm_syn.means, pipeline_gt_syn.means)
            gibbs_perm = find_optimal_permutation(pipeline_gibbs_syn.means, pipeline_gt_syn.means)

            results[0, i, j - 1] = find_mat_diff(pipeline_gt_syn.transmat,
                                                 reorient_matrix(pipeline_pome_syn.transmat, pome_perm)) / \
                                   pipeline_gt_syn.transmat.shape[0]
            """results[1, i, j - 1] = find_mat_diff(pipeline_gt_syn.transmat,
                                                 reorient_matrix(pipeline_hmm_syn.transmat, hmm_perm)) / \
                                   pipeline_gt_syn.transmat.shape[0]"""
            results[1, i, j - 1] = find_mat_diff(pipeline_gt_syn.transmat, pipeline_gibbs_syn.transmat) / \
                                   pipeline_gt_syn.transmat.shape[
                                       0]  # we don't change the orientation of the gibbs matrix

        x = [0.01]
        ax[j - 1].plot(x, results[0, :, j - 1], marker='D', label="pome", linestyle=":")
        ax[j - 1].plot(x, results[1, :, j - 1], marker='D', label="gibbs", linestyle=":")
        ax[j - 1].set_title(f"Run  #{j}")
        ax[j - 1].set_xscale('log')
        ax[j - 1].yaxis.set_major_locator(plt.MaxNLocator(5))

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower left')
    fig.subplots_adjust(hspace=0.5)
    fig.supxlabel("Sigma")
    fig.supylabel("L1 normalized")
    plt.show()
    plt.figure(10)
    x = [0.01]
    results = np.average(results, axis=2)
    plt.plot(x, results[0, :], marker='o', label="pome", linestyle=":")
    plt.plot(x, results[1, :], marker='o', label="gibbs", linestyle=":")
    plt.ylabel("L1 normalized")
    plt.xlabel("Sigma")
    plt.xscale("log")
    plt.title("Averaged Iterations")
    plt.legend()
    print(pipeline_gt_syn.reader.dataset['lengths'])


def compare_pipelines_vs_iter_pass_all(n_components, n_run, n_iter, show=True):
    reader = "My Synthetic"
    results = np.empty((2, n_run, n_iter + 1))
    means_results = np.empty((2, n_run, n_iter + 1, n_components))
    for j in range(n_run):
        pipeline_pome_syn: pipeline = create_pipeline(reader, "Pass All", "Pomegranate - Synthetic" + str(j + 1))
        print(f"finished creating {pipeline_pome_syn}")
        pipeline_gibbs_syn: pipeline = create_pipeline(reader, "Pass All", "Gibbs Sampler" + str(j + 1))
        print(f"finished creating {pipeline_gibbs_syn}")
        pipeline_gt_syn: pipeline = create_pipeline(reader, "Pass All", "Ground Truth" + str(j + 1))
        print(f"finished creating {pipeline_gt_syn}")

        print("matrices compared l1 norm")

        results[0, j, :] = np.array(
            compare_mat_l1_norm_for_list(pipeline_gt_syn.transmat_list, pipeline_pome_syn.transmat_list))
        means_results[0, j, :, :] = np.array(pipeline_pome_syn.means_list)
        results[1, j, :] = np.array(
            compare_mat_l1_norm_for_list(pipeline_gt_syn.transmat_list, pipeline_gibbs_syn.transmat_list))
        means_results[1, j, :, :] = np.array(pipeline_gibbs_syn.means_list)
        x = np.arange(n_iter + 1)
        plt.figure(1)
        plt.plot(x, results[0, j, :], marker='.', label=f"run #{j}", linestyle=":")
        plt.figure(2)
        plt.plot(x, results[1, j, :], marker='.', label=f"run #{j}", linestyle=":")

        plt.figure(3 + 2 * j)
        plt.plot(x, means_results[0, j, :, :], marker='.', label=f"run #{j}", linestyle="")
        plt.title(f"Pome run #{j}")
        plt.axhline(y=1, color='r', linestyle=':')
        plt.axhline(y=2, color='r', linestyle=':')
        plt.axhline(y=3, color='r', linestyle=':')
        plt.figure(4 + 2 * j)
        plt.plot(x, means_results[1, j, :, :], marker='.', label=f"run #{j}", linestyle="")
        plt.title(f"Gibbs run #{j}")
        print(np.sum(pipeline_gt_syn.reader.dataset['lengths']))

    plt.figure(1)
    plt.title("Pome")
    plt.xlabel("iter #")
    plt.ylabel("L1 Norm")
    plt.legend()
    plt.figure(2)
    plt.title("Gibbs")
    plt.xlabel("iter #")
    plt.ylabel("L1 Norm")
    plt.legend()

    if show:
        plt.show()


def compare_pipelines_for_different_prob(omitter, n_run, n_probabilities=None, probabilities=None, show=True):
    if n_probabilities is None:
        n_probabilities = len(probabilities)
    if probabilities is None:
        probabilities = np.linspace(0.1, 1, n_probabilities)
    results = np.empty((2, n_run, n_probabilities))
    reader = "My Synthetic"
    for j in range(n_run):
        for i, p in enumerate(probabilities):
            omitter_bernoulli_config: bernoulli_omitter_config
            reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
                reader, omitter,
                "Pomegranate - Synthetic" + str(j + 1))
            _, _, pipeline_gibbs_config = create_config(reader, "Bernoulli",
                                                        "Gibbs Sampler" + str(j + 1))
            _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                       "Ground Truth" + str(j + 1))

            omitter_bernoulli_config.prob_of_observation = p  # change the prob
            pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                       pipeline_pome_config)
            print(f"finished creating {pipeline_pome_syn}")
            pipeline_gibbs_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                        pipeline_gibbs_config)
            print(f"finished creating {pipeline_gibbs_syn}")
            pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                     pipeline_gt_config)
            print(f"finished creating {pipeline_gt_syn}")

            print("matrices compared l1 norm")
            results[0, j, i] = np.array(
                compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_pome_syn.transmat))
            results[1, j, i] = np.array(
                compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_gibbs_syn.transmat))
        plt.figure(1)
        plt.plot(probabilities, results[0, j, :], marker='.', label=f"run #{j}", linestyle=":")
        plt.figure(2)
        plt.plot(probabilities, results[1, j, :], marker='.', label=f"run #{j}", linestyle=":")

    plt.figure(1)
    plt.title("Pome " + omitter)
    plt.xlabel("probability of seeing emission")
    plt.ylabel("L1 Norm")
    plt.legend()
    plt.figure(2)
    plt.title("Gibbs " + omitter)
    plt.xlabel("probability of seeing emission")
    plt.ylabel("L1 Norm")
    plt.legend()

    if show:
        plt.show()


def compare_pipelines_for_different_prob_pome(omitter, n_run, n_probabilities=None, probabilities=None, show=True):
    if n_probabilities is None:
        n_probabilities = len(probabilities)
    if probabilities is None:
        probabilities = np.linspace(0.1, 1, n_probabilities)
    results = np.empty((n_run, n_probabilities))
    reader = "My Synthetic"
    for j in range(n_run):
        for i, p in enumerate(probabilities):
            omitter_bernoulli_config: bernoulli_omitter_config
            reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
                reader, omitter,
                "Pomegranate - Synthetic" + str(j + 1))
            _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                       "Ground Truth" + str(j + 1))

            omitter_bernoulli_config.prob_of_observation = p  # change the prob
            pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                       pipeline_pome_config)
            print(f"finished creating {pipeline_pome_syn}")
            pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                     pipeline_gt_config)
            print(f"finished creating {pipeline_gt_syn}")

            print("matrices compared l1 norm")
            results[j, i] = np.array(
                compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_pome_syn.transmat))
        plt.figure(1)
        plt.plot(probabilities, results[j, :], marker='.', label=f"run #{j}", linestyle=":")

    plt.figure(1)
    plt.title("Pome " + omitter)
    plt.xlabel("probability of seeing emission")
    plt.ylabel("L1 Norm")
    plt.legend()

    if show:
        plt.show()


def compare_pipelines_vs_iter_pass_all_different_sample_len(n_run, n_iter, n_samples_arr, sentence_length_arr,
                                                            n_components=10, show=True):
    n_sample_len = len(n_samples_arr)
    if len(sentence_length_arr) != n_sample_len:
        raise ValueError(
            f"Sentence Length Array:{sentence_length_arr} does not have the same number of elements as the Number of Samples Array: {n_samples_arr} ")
    reader = "My Synthetic"
    results = np.empty((2, n_sample_len, n_run, n_iter + 1))
    for i, (n_samples, sentence_length) in enumerate(zip(n_samples_arr, sentence_length_arr)):
        for j in range(n_run):
            reader_config: my_synthetic_reader_config
            reader_config, omitter_config, pipeline_pome_config = create_config(reader, "Pass All",
                                                                                "Pomegranate - Synthetic" + str(j + 1))
            _, _, pipeline_gibbs_config = create_config(reader, "Pass All",
                                                        "Gibbs Sampler" + str(j + 1))
            _, _, pipeline_gt_config = create_config(reader, "Pass All",
                                                     "Ground Truth" + str(j + 1))

            reader_config.n_samples = n_samples
            reader_config.sentence_length = sentence_length
            pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_config,
                                                                       pipeline_pome_config)
            print(f"finished creating {pipeline_pome_syn}")
            pipeline_gibbs_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_config,
                                                                        pipeline_gibbs_config)
            print(f"finished creating {pipeline_gibbs_syn}")
            pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_config,
                                                                     pipeline_gt_config)
            print(f"finished creating {pipeline_gt_syn}")
            print("matrices compared l1 norm")

            results[0, i, j] = np.array(
                compare_mat_l1_norm_for_list(pipeline_gt_syn.transmat_list, pipeline_pome_syn.transmat_list))
            results[1, i, j] = np.array(
                compare_mat_l1_norm_for_list(pipeline_gt_syn.transmat_list, pipeline_gibbs_syn.transmat_list))
            x = np.arange(n_iter + 1)
            plt.figure(2 * i)
            plt.plot(x, results[0, i, j], marker='.', label=f"run #{j}", linestyle=":")
            plt.figure(2 * i + 1)
            plt.plot(x, results[1, i, j], marker='.', label=f"run #{j}", linestyle=":")
        plt.figure(2 * i)
        plt.title(f"Pome - Sentence Length:{sentence_length}, Number of Samples: {n_samples}")
        plt.xlabel("iter #")
        plt.ylabel("L1 Norm")
        plt.legend()
        plt.figure(2 * i + 1)
        plt.title(f"Gibbs - Sentence Length:{sentence_length}, Number of Samples: {n_samples}")
        plt.plot(x, results[1, i, j], marker='.', label=f"run #{j}", linestyle=":")
        plt.xlabel("iter #")
        plt.ylabel("L1 Norm")
        plt.legend()

    if show:
        plt.show()


def compare_pipelines_for_different_prob_vs_iter(omitter, n_run, n_iter, n_probabilities=None,
                                                 probabilities=None, show=True):
    if n_probabilities is None:
        n_probabilities = len(probabilities)
    if probabilities is None:
        probabilities = np.linspace(0.1, 1, n_probabilities)
    results = np.empty((2, n_run, n_probabilities, n_iter + 1))

    fig1, axes1 = plt.subplots(n_probabilities, 1, sharex=True, sharey=True, figsize=(10, 6))

    fig2, axes2 = plt.subplots(n_probabilities, 1, sharex=True, sharey=True, figsize=(10, 6))
    reader = "My Synthetic"
    for j in range(n_run):
        for i, p in enumerate(probabilities):
            omitter_bernoulli_config: bernoulli_omitter_config
            reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
                reader, omitter,
                "Pomegranate - Synthetic" + str(j + 1))
            _, _, pipeline_gibbs_config = create_config(reader, "Bernoulli",
                                                        "Gibbs Sampler" + str(j + 1))
            _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                       "Ground Truth" + str(j + 1))

            omitter_bernoulli_config.prob_of_observation = p  # change the prob
            pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                       pipeline_pome_config)
            print(f"finished creating {pipeline_pome_syn}")
            pipeline_gibbs_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                        pipeline_gibbs_config)
            print(f"finished creating {pipeline_gibbs_syn}")
            pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                     pipeline_gt_config)
            print(f"finished creating {pipeline_gt_syn}")

            print("matrices compared l1 norm")
            results[0, j, i] = np.array(
                compare_mat_l1_norm_for_list(pipeline_gt_syn.transmat_list, pipeline_pome_syn.transmat_list))
            results[1, j, i] = np.array(
                compare_mat_l1_norm_for_list(pipeline_gt_syn.transmat_list, pipeline_gibbs_syn.transmat_list))
            x = np.arange(n_iter + 1)

            axes1[i].plot(x, results[0, j, i], marker='.', label=f"run #{j}", linestyle=":")
            axes1[i].set_title(f"p={p}")

            axes2[i].plot(x, results[1, j, i], marker='.', label=f"run #{j}", linestyle=":")
            axes2[i].set_title(f"p={p}")

    handles, labels = axes1[0].get_legend_handles_labels()
    fig1.legend(handles, labels, loc='lower left')
    fig1.suptitle("Pome " + omitter)
    fig1.subplots_adjust(hspace=0.6)
    fig1.supxlabel("Iter #")
    handles, labels = axes2[0].get_legend_handles_labels()
    fig2.legend(handles, labels, loc='lower left')
    fig2.suptitle("Gibbs " + omitter)
    fig2.subplots_adjust(hspace=0.6)
    fig2.supxlabel("Iter #")
    if show:
        plt.show()


def compare_pipelines_for_different_prob_vs_iter(omitter, n_run, n_iter, n_probabilities=None,
                                                 probabilities=None, show=True):
    if n_probabilities is None:
        n_probabilities = len(probabilities)
    if probabilities is None:
        probabilities = np.linspace(0.1, 1, n_probabilities)
    results = np.empty((n_run, n_probabilities, n_iter + 1))

    fig1, axes1 = plt.subplots(n_probabilities, 1, sharex=True, sharey=True, figsize=(10, 6))

    reader = "My Synthetic"
    for j in range(n_run):
        for i, p in enumerate(probabilities):
            omitter_bernoulli_config: bernoulli_omitter_config
            reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
                reader, omitter,
                "Pomegranate - Synthetic" + str(j + 1))
            _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                       "Ground Truth" + str(j + 1))

            omitter_bernoulli_config.prob_of_observation = p  # change the prob
            pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                       pipeline_pome_config)
            print(f"finished creating {pipeline_pome_syn}")
            pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                     pipeline_gt_config)
            print(f"finished creating {pipeline_gt_syn}")

            print("matrices compared l1 norm")
            results[j, i] = np.array(
                compare_mat_l1_norm_for_list(pipeline_gt_syn.transmat_list, pipeline_pome_syn.transmat_list))
            x = np.arange(n_iter + 1)

            axes1[i].plot(x, results[j, i], marker='.', label=f"run #{j}", linestyle=":")
            axes1[i].set_title(f"p={p}")

    handles, labels = axes1[0].get_legend_handles_labels()
    fig1.legend(handles, labels, loc='lower left')
    fig1.suptitle("Pome " + omitter)
    fig1.subplots_adjust(hspace=0.6)
    fig1.supxlabel("Iter #")
    if show:
        plt.show()


def compare_pipelines_for_different_prob_vs_temporal_info(omitter, n_run, powers, n_probabilities=None,
                                                          probabilities=None, show=True):
    if n_probabilities is None:
        n_probabilities = len(probabilities)
    if probabilities is None:
        probabilities = np.linspace(0.1, 1, n_probabilities)
    results = np.empty((2, n_run, n_probabilities, len(powers)))

    fig1, axes1 = plt.subplots(n_probabilities, 1, sharex=True, sharey=True, figsize=(10, 6))

    fig2, axes2 = plt.subplots(n_probabilities, 1, sharex=True, sharey=True, figsize=(10, 6))
    reader = "My Synthetic"
    for j in range(n_run):
        omitter_bernoulli_config: bernoulli_omitter_config
        reader_config: my_synthetic_reader_config
        reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
            reader, omitter,
            "Pomegranate - Synthetic" + str(j + 1))
        _, _, pipeline_gibbs_config = create_config(reader, "Bernoulli",
                                                    "Gibbs Sampler" + str(j + 1))
        _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                   "Ground Truth" + str(j + 1))

        for i, p in enumerate(probabilities):
            omitter_bernoulli_config.prob_of_observation = p  # change the prob
            x = []
            for k, power in enumerate(powers):
                reader_config.set_temporal = True
                reader_config.matrix_power = power

                pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                           pipeline_pome_config)
                print(f"finished creating {pipeline_pome_syn}")
                pipeline_gibbs_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                            pipeline_gibbs_config)
                print(f"finished creating {pipeline_gibbs_syn}")
                pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                         pipeline_gt_config)
                print(f"finished creating {pipeline_gt_syn}")

                print("matrices compared l1 norm")
                results[0, j, i, k - 1] = compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_pome_syn.transmat)
                results[1, j, i, k - 1] = compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_gibbs_syn.transmat)
                x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

            axes1[i].plot(x, results[0, j, i, :], marker='.', label=f"run #{j}", linestyle=":")
            axes1[i].set_title(f"p={p}")

            axes2[i].plot(x, results[1, j, i, :], marker='.', label=f"run #{j}", linestyle=":")
            axes2[i].set_title(f"p={p}")

    handles, labels = axes1[0].get_legend_handles_labels()
    fig1.legend(handles, labels, loc='lower left')
    fig1.suptitle("Pome " + omitter)
    fig1.subplots_adjust(hspace=0.6)
    fig1.supxlabel("Temporal Info Ratio")
    handles, labels = axes2[0].get_legend_handles_labels()
    fig2.legend(handles, labels, loc='lower left')
    fig2.suptitle("Gibbs " + omitter)
    fig2.subplots_adjust(hspace=0.6)
    fig2.supxlabel("Temporal Info Ratio")
    if show:
        plt.show()


def compare_pipelines_for_different_prob_transmat_mode_vs_temporal_info_pome(omitter, n_run, powers, transmat_mode=None,
                                                                             n_probabilities=None, probabilities=None,
                                                                             show=True):
    if n_probabilities is None:
        n_probabilities = len(probabilities)
    if probabilities is None:
        probabilities = np.linspace(0.1, 1, n_probabilities)
    results = np.empty((n_run, n_probabilities, len(powers)))

    fig1, axes1 = plt.subplots(n_probabilities, 1, sharex=True, sharey=True, figsize=(10, 6))

    reader = "My Synthetic"
    for j in range(n_run):
        omitter_bernoulli_config: bernoulli_omitter_config
        reader_config: my_synthetic_reader_config
        reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
            reader, omitter,
            "Pomegranate - Synthetic" + str(j + 1))
        _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                   "Ground Truth" + str(j + 1))
        reader_config.transmat_mode = transmat_mode

        for i, p in enumerate(probabilities):
            omitter_bernoulli_config.prob_of_observation = p  # change the prob
            x = []
            for k, power in enumerate(powers):
                reader_config.set_temporal = True
                reader_config.matrix_power = power

                pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                           pipeline_pome_config)
                print(f"finished creating {pipeline_pome_syn}")
                pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                         pipeline_gt_config)
                print(f"finished creating {pipeline_gt_syn}")

                print("matrices compared l1 norm")
                results[j, i, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_pome_syn.transmat)
                x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

            axes1[i].plot(x, results[j, i, :], marker='.', label=f"run #{j}", linestyle=":")
            axes1[i].set_title(f"p={p}")

    handles, labels = axes1[0].get_legend_handles_labels()
    fig1.legend(handles, labels, loc='lower left')
    fig1.suptitle("Pome " + omitter + " " + transmat_mode)
    fig1.subplots_adjust(hspace=0.6)
    fig1.supxlabel("Temporal Info Ratio")
    if show:
        plt.show()


def compare_pipelines_for_different_prob_transmat_mode_vs_temporal_info(omitter, n_run, powers, transmat_mode=None,
                                                                        n_probabilities=None, probabilities=None,
                                                                        show=True):
    if n_probabilities is None:
        n_probabilities = len(probabilities)
    if probabilities is None:
        probabilities = np.linspace(0.1, 1, n_probabilities)
    results = np.empty((2, n_run, n_probabilities, len(powers)))

    fig1, axes1 = plt.subplots(n_probabilities, 1, sharex=True, sharey=True, figsize=(10, 6))

    fig2, axes2 = plt.subplots(n_probabilities, 1, sharex=True, sharey=True, figsize=(10, 6))
    reader = "My Synthetic"
    for j in range(n_run):
        omitter_bernoulli_config: bernoulli_omitter_config
        reader_config: my_synthetic_reader_config
        reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
            reader, omitter,
            "Pomegranate - Synthetic" + str(j + 1))
        _, _, pipeline_gibbs_config = create_config(reader, "Bernoulli",
                                                    "Gibbs Sampler" + str(j + 1))
        _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                   "Ground Truth" + str(j + 1))
        reader_config.transmat_mode = transmat_mode

        for i, p in enumerate(probabilities):
            omitter_bernoulli_config.prob_of_observation = p  # change the prob
            x = []
            for k, power in enumerate(powers):
                reader_config.set_temporal = True
                reader_config.matrix_power = power

                pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                           pipeline_pome_config)
                print(f"finished creating {pipeline_pome_syn}")
                pipeline_gibbs_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                            pipeline_gibbs_config)
                print(f"finished creating {pipeline_gibbs_syn}")
                pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                         pipeline_gt_config)
                print(f"finished creating {pipeline_gt_syn}")

                print("matrices compared l1 norm")
                results[0, j, i, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_pome_syn.transmat)
                results[1, j, i, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_gibbs_syn.transmat)
                x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

            axes1[i].plot(x, results[0, j, i, :], marker='.', label=f"run #{j}", linestyle=":")
            axes1[i].set_title(f"p={p}")

            axes2[i].plot(x, results[1, j, i, :], marker='.', label=f"run #{j}", linestyle=":")
            axes2[i].set_title(f"p={p}")

    handles, labels = axes1[0].get_legend_handles_labels()
    fig1.legend(handles, labels, loc='lower left')
    fig1.suptitle("Pome " + omitter + " " + transmat_mode)
    fig1.subplots_adjust(hspace=0.6)
    fig1.supxlabel("Temporal Info Ratio")
    handles, labels = axes2[0].get_legend_handles_labels()
    fig2.legend(handles, labels, loc='lower left')
    fig2.suptitle("Gibbs " + omitter + " " + transmat_mode)
    fig2.subplots_adjust(hspace=0.6)
    fig2.supxlabel("Temporal Info Ratio")
    if show:
        plt.show()


def run_simple_test(reader_name, omitter_name, pipeline_name):
    fitted_pipeline: pipeline = create_pipeline(reader_name, omitter_name, pipeline_name)
    print(f"finished creating {fitted_pipeline}")
    print(fitted_pipeline.transmat)


def run_simple_test_transmat_mode(reader_name, omitter_name, pipeline_name, transmat_mode):
    reader_config, omitter_config, pipeline_config = create_config(reader_name, omitter_name, pipeline_name)
    reader_config.transmat_mode = transmat_mode
    fitted_pipeline: pipeline = create_pipeline_from_configs(reader_config, omitter_config, pipeline_config)
    print(f"finished creating {fitted_pipeline}")
    print(fitted_pipeline.transmat)


def compare_pipelines_for_different_prob_transmat_mode_with_temp_info_vs_prob(omitter, n_run, powers,
                                                                              transmat_mode=None,
                                                                              n_probabilities=None, probabilities=None,
                                                                              show=True):
    if n_probabilities is None:
        n_probabilities = len(probabilities)
    if probabilities is None:
        probabilities = np.linspace(0.1, 1, n_probabilities)
    results = np.empty((2, n_run, n_probabilities, len(powers)))
    reader = "My Synthetic"
    for j in range(n_run):
        omitter_bernoulli_config: bernoulli_omitter_config
        reader_config: my_synthetic_reader_config
        reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
            reader, omitter,
            "Pomegranate - Synthetic" + str(j + 1))
        _, _, pipeline_gibbs_config = create_config(reader, omitter,
                                                    "Gibbs Sampler" + str(j + 1))
        _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                   "Ground Truth" + str(j + 1))
        reader_config.transmat_mode = transmat_mode

        for i, p in enumerate(probabilities):
            omitter_bernoulli_config.prob_of_observation = p  # change the prob
            x = []
            for k, power in enumerate(powers):
                reader_config.set_temporal = True
                reader_config.matrix_power = power

                pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                           pipeline_pome_config)
                print(f"finished creating {pipeline_pome_syn}")
                pipeline_gibbs_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                            pipeline_gibbs_config)
                print(f"finished creating {pipeline_gibbs_syn}")
                pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                         pipeline_gt_config)
                print(f"finished creating {pipeline_gt_syn}")

                print("matrices compared l1 norm")
                results[0, j, i, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat,
                                                          pipeline_pome_syn.transmat)
                results[1, j, i, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat,
                                                          pipeline_gibbs_syn.transmat)
                x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

    results = np.average(results, axis=1)
    cmap = cm.get_cmap('plasma')
    for k, power in enumerate(powers):
        color = cmap(x[k])
        plt.figure(1)
        plt.plot(probabilities, results[0, :, k], marker='.', label=f"temp info = {x[k]:.2f}", linestyle=":",
                 color=color)

        plt.figure(2)
        plt.plot(probabilities, results[1, :, k], marker='.', label=f"temp info = {x[k]:.2f}", linestyle=":",
                 color=color)

    plt.figure(1)
    plt.legend()
    plt.title("Pome " + omitter + " " + transmat_mode)
    plt.xlabel("probability")
    plt.ylabel("L1")
    plt.figure(2)
    plt.legend()
    plt.title("Gibbs " + omitter + " " + transmat_mode)
    plt.xlabel("probability")
    plt.ylabel("L1")
    if show:
        plt.show()


def compare_pipelines_for_different_prob_transmat_mode_vs_temporal_info(n_run, powers, transmat_modes, show=True):
    omitter = "Bernoulli"
    reader = "My Synthetic"
    for i, transmat_mode in enumerate(transmat_modes):
        results = np.empty((2, n_run, len(powers[i])))
        for j in range(n_run):
            omitter_bernoulli_config: bernoulli_omitter_config
            reader_config: my_synthetic_reader_config
            reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
                reader, omitter,
                "Pomegranate - Synthetic" + str(j + 1))
            _, _, pipeline_gibbs_config = create_config(reader, "Bernoulli",
                                                        "Gibbs Sampler" + str(j + 1))
            _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                       "Ground Truth" + str(j + 1))
            reader_config.transmat_mode = transmat_mode
            omitter_bernoulli_config.prob_of_observation = 1.0  # pass all
            x = []
            for k, power in enumerate(powers[i]):
                reader_config.set_temporal = True
                reader_config.matrix_power = power

                pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                           pipeline_pome_config)
                print(f"finished creating {pipeline_pome_syn}")
                pipeline_gibbs_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                            pipeline_gibbs_config)
                print(f"finished creating {pipeline_gibbs_syn}")
                pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                         pipeline_gt_config)
                print(f"finished creating {pipeline_gt_syn}")

                print("matrices compared l1 norm")

                results[0, j, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_pome_syn.transmat)
                results[1, j, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_gibbs_syn.transmat)
                x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

        results = np.average(results, axis=1)
        plt.figure(1, figsize=(height, width))
        plt.plot(x, results[0], marker='.', label=f"Transition Mode = {transmat_mode}", linestyle=":")

        plt.figure(2, figsize=(height, width))
        plt.plot(x, results[1], marker='.', label=f"Transition Mode = {transmat_mode}", linestyle=":")

    plt.figure(1)
    plt.legend()
    plt.title("Pome Pass All")
    plt.xlabel("Temporal Info")
    plt.ylabel("L1")
    plt.savefig("Pome Pass All")
    plt.figure(2)
    plt.legend()
    plt.title("Gibbs Pass All")
    plt.xlabel("Temporal Info")
    plt.ylabel("L1")
    plt.savefig("Gibbs Pass All")
    if show:
        plt.show()


def compare_pipelines_for_different_prob_transmat_mode_vs_temporal_info_norm(n_run, powers, transmat_modes, show=True):
    omitter = "Bernoulli"
    reader = "My Synthetic"
    for i, transmat_mode in enumerate(transmat_modes):
        results = np.empty((2, n_run, len(powers[i])))
        for j in range(n_run):
            omitter_bernoulli_config: bernoulli_omitter_config
            reader_config: my_synthetic_reader_config
            reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
                reader, omitter,
                "Pomegranate - Synthetic" + str(j + 1))
            _, _, pipeline_gibbs_config = create_config(reader, "Bernoulli",
                                                        "Gibbs Sampler" + str(j + 1))
            _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                       "Ground Truth" + str(j + 1))
            reader_config.transmat_mode = transmat_mode
            omitter_bernoulli_config.prob_of_observation = 1.0  # pass all
            x = []
            for k, power in enumerate(powers[i]):
                reader_config.set_temporal = True
                reader_config.matrix_power = power

                pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                           pipeline_pome_config)
                print(f"finished creating {pipeline_pome_syn}")
                pipeline_gibbs_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                            pipeline_gibbs_config)
                print(f"finished creating {pipeline_gibbs_syn}")
                pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                         pipeline_gt_config)
                print(f"finished creating {pipeline_gt_syn}")

                print("matrices compared l1 norm")
                pome_stochastic_value = compare_mat_l1_norm(pipeline_gt_syn.transmat_list[0],
                                                            pipeline_pome_syn.transmat_list[0])
                gibbs_stochastic_value = compare_mat_l1_norm(pipeline_gt_syn.transmat_list[0],
                                                             pipeline_gibbs_syn.transmat_list[0])
                results[0, j, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_pome_syn.transmat) / \
                                   pome_stochastic_value
                results[1, j, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_gibbs_syn.transmat) / \
                                   gibbs_stochastic_value
                x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

        results = np.average(results, axis=1)
        plt.figure(1, figsize=(height, width))
        plt.plot(x, results[0], marker='.', label=f"Transition Mode = {transmat_mode}", linestyle=":")

        plt.figure(2, figsize=(height, width))
        plt.plot(x, results[1], marker='.', label=f"Transition Mode = {transmat_mode}", linestyle=":")

    plt.figure(1)
    plt.legend()
    plt.title("Pome Pass All")
    plt.xlabel("Temporal Info")
    plt.ylabel("L1")
    plt.savefig("Pome Pass All" + "Norm")
    plt.figure(2)
    plt.legend()
    plt.title("Gibbs Pass All")
    plt.xlabel("Temporal Info")
    plt.ylabel("L1")
    plt.savefig("Gibbs Pass All" + "Norm")
    if show:
        plt.show()


def compare_results_from_different_omitters(n_run, powers, transmat_mode, omitters, probabilities, show=True):
    reader = "My Synthetic"
    results = np.empty((2, n_run, len(probabilities), len(powers)))
    for i, omitter in enumerate(omitters):
        for j in range(n_run):
            reader_config: my_synthetic_reader_config
            reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
                reader, omitter,
                "Pomegranate - Synthetic" + str(j + 1))
            _, _, pipeline_gibbs_config = create_config(reader, omitter,
                                                        "Gibbs Sampler" + str(j + 1))
            _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                       "Ground Truth" + str(j + 1))
            reader_config.transmat_mode = transmat_mode
            omitter_bernoulli_config.prob_of_observation = probabilities[i]
            x = []
            for k, power in enumerate(powers):
                reader_config.set_temporal = True
                reader_config.matrix_power = power

                pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                           pipeline_pome_config)
                print(f"finished creating {pipeline_pome_syn}")
                pipeline_gibbs_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                            pipeline_gibbs_config)
                print(f"finished creating {pipeline_gibbs_syn}")
                pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                         pipeline_gt_config)
                print(f"finished creating {pipeline_gt_syn}")

                print("matrices compared l1 norm")
                results[0, j, i, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_pome_syn.transmat)
                results[1, j, i, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_gibbs_syn.transmat)
                x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

        averaged_results = np.average(results, axis=1)
        plt.figure(1, figsize=(height, width))
        plt.plot(x, averaged_results[0, i], marker='.', label=f"Omitter = {omitter}", linestyle=":")

        plt.figure(2, figsize=(height, width))
        plt.plot(x, averaged_results[1, i], marker='.', label=f"Omitter = {omitter}", linestyle=":")

    plt.figure(1)
    plt.legend()
    plt.title("Pome Half Seen " + transmat_mode)
    plt.xlabel("Temporal Info")
    plt.ylabel("L1")
    plt.figure(2)
    plt.legend()
    plt.title("Gibbs Half Seen " + transmat_mode)
    plt.xlabel("Temporal Info")
    plt.ylabel("L1")
    if show:
        plt.show()


def compare_results_from_different_omitters_transmat_modes(n_run, powers, transmat_modes, omitters, probabilities,
                                                           show=True):
    reader = "My Synthetic"
    fig1, axes1 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))
    fig2, axes2 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))

    for l, transmat_mode in enumerate(transmat_modes):
        results = np.empty((2, n_run, len(probabilities), len(powers[l])))

        for i, omitter in enumerate(omitters):
            for j in range(n_run):
                reader_config: my_synthetic_reader_config
                reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
                    reader, omitter,
                    "Pomegranate - Synthetic" + str(j + 1))
                _, _, pipeline_gibbs_config = create_config(reader, omitter,
                                                            "Gibbs Sampler" + str(j + 1))
                _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                           "Ground Truth" + str(j + 1))
                reader_config.transmat_mode = transmat_mode
                omitter_bernoulli_config.prob_of_observation = probabilities[i]
                x = []
                for k, power in enumerate(powers[l]):
                    reader_config.set_temporal = True
                    reader_config.matrix_power = power

                    pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                               pipeline_pome_config)
                    print(f"finished creating {pipeline_pome_syn}")
                    pipeline_gibbs_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                                pipeline_gibbs_config)
                    print(f"finished creating {pipeline_gibbs_syn}")
                    pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                             pipeline_gt_config)
                    print(f"finished creating {pipeline_gt_syn}")

                    print("matrices compared l1 norm")
                    results[0, j, i, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_pome_syn.transmat)
                    results[1, j, i, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_gibbs_syn.transmat)
                    x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

        averaged_results = np.average(results, axis=1)
        for i, omitter in enumerate(omitters):
            axes1[l].plot(x, averaged_results[0, i], marker='.', label=f"Omitter = {omitter}", linestyle=":")
            axes1[l].set_ylabel("L1")
            axes1[l].set_title(transmat_mode)

            axes2[l].plot(x, averaged_results[1, i], marker='.', label=f"Omitter = {omitter}", linestyle=":")
            axes2[l].set_ylabel("L1")
            axes2[l].set_title(transmat_mode)
        axes1[l].legend()
        axes2[l].legend()

    fig1.suptitle("Pome 50 Percent")
    fig1.supxlabel("temporal information")
    fig1.savefig("Pome 50 Percent")
    fig2.suptitle("Gibbs 50 Percent")
    fig2.supxlabel("temporal information")
    fig2.savefig("Gibbs 50 Percent")
    if show:
        plt.show()


def compare_results_from_different_omitters_transmat_modes_norm(n_run, powers, transmat_modes, omitters, probabilities,
                                                                show=True):
    reader = "My Synthetic"
    fig1, axes1 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))
    fig2, axes2 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))

    for l, transmat_mode in enumerate(transmat_modes):
        results = np.empty((2, n_run, len(probabilities), len(powers[l])))

        for i, omitter in enumerate(omitters):
            for j in range(n_run):
                reader_config: my_synthetic_reader_config
                reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
                    reader, omitter,
                    "Pomegranate - Synthetic" + str(j + 1))
                _, _, pipeline_gibbs_config = create_config(reader, omitter,
                                                            "Gibbs Sampler" + str(j + 1))
                _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                           "Ground Truth" + str(j + 1))
                reader_config.transmat_mode = transmat_mode
                omitter_bernoulli_config.prob_of_observation = probabilities[i]
                x = []
                for k, power in enumerate(powers[l]):
                    reader_config.set_temporal = True
                    reader_config.matrix_power = power

                    pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                               pipeline_pome_config)
                    print(f"finished creating {pipeline_pome_syn}")
                    pipeline_gibbs_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                                pipeline_gibbs_config)
                    print(f"finished creating {pipeline_gibbs_syn}")
                    pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                             pipeline_gt_config)
                    print(f"finished creating {pipeline_gt_syn}")

                    print("matrices compared l1 norm")
                    pome_stochastic_value = compare_mat_l1_norm(pipeline_gt_syn.transmat_list[0],
                                                                pipeline_pome_syn.transmat_list[0])
                    gibbs_stochastic_value = compare_mat_l1_norm(pipeline_gt_syn.transmat_list[0],
                                                                 pipeline_gibbs_syn.transmat_list[0])
                    results[0, j, i, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_pome_syn.transmat) / \
                                          pome_stochastic_value
                    results[1, j, i, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_gibbs_syn.transmat) / \
                                          gibbs_stochastic_value
                    x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

        averaged_results = np.average(results, axis=1)
        for i, omitter in enumerate(omitters):
            axes1[l].plot(x, averaged_results[0, i], marker='.', label=f"Omitter = {omitter}", linestyle=":")
            axes1[l].set_ylabel("L1")
            axes1[l].set_title(transmat_mode)

            axes2[l].plot(x, averaged_results[1, i], marker='.', label=f"Omitter = {omitter}", linestyle=":")
            axes2[l].set_ylabel("L1")
            axes2[l].set_title(transmat_mode)
        axes1[l].legend()
        axes2[l].legend()

    fig1.suptitle("Pome 50 Percent")
    fig1.supxlabel("temporal information")
    fig1.savefig("Pome 50 Percent" + "Norm")
    fig2.suptitle("Gibbs 50 Percent")
    fig2.supxlabel("temporal information")
    fig2.savefig("Gibbs 50 Percent" + "Norm")
    if show:
        plt.show()


def compare_pipelines_for_different_prob_transmat_modes_with_temp_info_vs_prob(omitter, n_run, powers,
                                                                               transmat_modes=None,
                                                                               n_probabilities=None, probabilities=None,
                                                                               show=True):
    fig1, axes1 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))
    fig2, axes2 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))
    for l, transmat_mode in enumerate(transmat_modes):
        if n_probabilities is None:
            n_probabilities = len(probabilities)
        if probabilities is None:
            probabilities = np.linspace(0.1, 1, n_probabilities)
        results = np.empty((2, n_run, n_probabilities, len(powers[l])))
        reader = "My Synthetic"
        for j in range(n_run):
            omitter_bernoulli_config: bernoulli_omitter_config
            reader_config: my_synthetic_reader_config
            reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
                reader, omitter,
                "Pomegranate - Synthetic" + str(j + 1))
            _, _, pipeline_gibbs_config = create_config(reader, "Bernoulli",
                                                        "Gibbs Sampler" + str(j + 1))
            _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                       "Ground Truth" + str(j + 1))
            reader_config.transmat_mode = transmat_mode

            for i, p in enumerate(probabilities):
                omitter_bernoulli_config.prob_of_observation = p  # change the prob
                x = []
                for k, power in enumerate(powers[l]):
                    reader_config.set_temporal = True
                    reader_config.matrix_power = power

                    pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                               pipeline_pome_config)
                    print(f"finished creating {pipeline_pome_syn}")
                    pipeline_gibbs_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                                pipeline_gibbs_config)
                    print(f"finished creating {pipeline_gibbs_syn}")
                    pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                             pipeline_gt_config)
                    print(f"finished creating {pipeline_gt_syn}")

                    print("matrices compared l1 norm")
                    results[0, j, i, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat,
                                                              pipeline_pome_syn.transmat)
                    results[1, j, i, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat,
                                                              pipeline_gibbs_syn.transmat)
                    x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

        results = np.average(results, axis=1)
        cmap = cm.get_cmap('viridis')
        for k, power in enumerate(powers[l]):
            color = cmap(x[k] / max(x))  # Normalize x[k] to [0, 1]
            axes1[l].plot(probabilities, results[0, :, k], marker='.', label=f"temp info = {x[k]:.2f}", linestyle=":",
                          color=color)
            axes1[l].set_title(transmat_mode)
            axes1[l].set_ylabel("L1")

            axes2[l].plot(probabilities, results[1, :, k], marker='.', label=f"temp info = {x[k]:.2f}", linestyle=":",
                          color=color)
            axes2[l].set_title(transmat_mode)
            axes2[l].set_ylabel("L1")
        axes1[l].legend()
        axes2[l].legend()

    fig1.suptitle("Pome " + omitter)
    fig1.supxlabel("Probability")
    fig1.savefig("Pome" + omitter)

    fig2.suptitle("Gibbs " + omitter)
    fig2.supxlabel("Probability")
    fig2.savefig("Gibbs" + omitter)

    if show:
        plt.show()


def compare_pipelines_for_different_prob_transmat_modes_with_temp_info_vs_prob_norm(omitter, n_run, powers,
                                                                                    transmat_modes=None,
                                                                                    n_probabilities=None,
                                                                                    probabilities=None,
                                                                                    show=True):
    fig1, axes1 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))
    fig2, axes2 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))
    for l, transmat_mode in enumerate(transmat_modes):
        if n_probabilities is None:
            n_probabilities = len(probabilities)
        if probabilities is None:
            probabilities = np.linspace(0.1, 1, n_probabilities)
        results = np.empty((2, n_run, n_probabilities, len(powers[l])))
        reader = "My Synthetic"
        for j in range(n_run):
            omitter_bernoulli_config: bernoulli_omitter_config
            reader_config: my_synthetic_reader_config
            reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
                reader, omitter,
                "Pomegranate - Synthetic" + str(j + 1))
            _, _, pipeline_gibbs_config = create_config(reader, "Bernoulli",
                                                        "Gibbs Sampler" + str(j + 1))
            _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                       "Ground Truth" + str(j + 1))
            reader_config.transmat_mode = transmat_mode

            for i, p in enumerate(probabilities):
                omitter_bernoulli_config.prob_of_observation = p  # change the prob
                x = []
                for k, power in enumerate(powers[l]):
                    reader_config.set_temporal = True
                    reader_config.matrix_power = power

                    pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                               pipeline_pome_config)
                    print(f"finished creating {pipeline_pome_syn}")
                    pipeline_gibbs_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                                pipeline_gibbs_config)
                    print(f"finished creating {pipeline_gibbs_syn}")
                    pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                             pipeline_gt_config)
                    print(f"finished creating {pipeline_gt_syn}")

                    print("matrices compared l1 norm")
                    pome_stochastic_value = compare_mat_l1_norm(pipeline_gt_syn.transmat_list[0],
                                                                pipeline_pome_syn.transmat_list[0])
                    gibbs_stochastic_value = compare_mat_l1_norm(pipeline_gt_syn.transmat_list[0],
                                                                 pipeline_gibbs_syn.transmat_list[0])
                    results[0, j, i, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat,
                                                              pipeline_pome_syn.transmat) / pome_stochastic_value
                    results[1, j, i, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat,
                                                              pipeline_gibbs_syn.transmat) / gibbs_stochastic_value
                    x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

        results = np.average(results, axis=1)
        cmap = cm.get_cmap('viridis')
        for k, power in enumerate(powers[l]):
            color = cmap(x[k] / max(x))  # Normalize x[k] to [0, 1]
            axes1[l].plot(probabilities, results[0, :, k], marker='.', label=f"temp info = {x[k]:.2f}", linestyle=":",
                          color=color)
            axes1[l].set_title(transmat_mode)
            axes1[l].set_ylabel("L1")

            axes2[l].plot(probabilities, results[1, :, k], marker='.', label=f"temp info = {x[k]:.2f}", linestyle=":",
                          color=color)
            axes2[l].set_title(transmat_mode)
            axes2[l].set_ylabel("L1")
        axes1[l].legend()
        axes2[l].legend()

    fig1.suptitle("Pome " + omitter)
    fig1.supxlabel("Probability")
    fig1.savefig("Pome" + omitter + "Norm")

    fig2.suptitle("Gibbs " + omitter)
    fig2.supxlabel("Probability")
    fig2.savefig("Gibbs" + omitter + "Norm")

    if show:
        plt.show()


def compare_pipelines_transmat_mode_with_vs_temp_info_var(omitter, n_run, powers,
                                                          transmat_modes=None,
                                                          show=True):
    fig1, axes1 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))
    fig2, axes2 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))
    for l, transmat_mode in enumerate(transmat_modes):
        results = np.empty((2, n_run, len(powers[l])))
        reader = "My Synthetic"
        for j in range(n_run):
            reader_config: my_synthetic_reader_config
            reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
                reader, omitter,
                "Pomegranate - Synthetic" + str(j + 1))
            _, _, pipeline_gibbs_config = create_config(reader, omitter,
                                                        "Gibbs Sampler" + str(j + 1))
            _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                       "Ground Truth" + str(j + 1))
            reader_config.transmat_mode = transmat_mode

            x = []
            for k, power in enumerate(powers[l]):
                reader_config.set_temporal = True
                reader_config.matrix_power = power

                pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                           pipeline_pome_config)
                print(f"finished creating {pipeline_pome_syn}")
                pipeline_gibbs_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                            pipeline_gibbs_config)
                print(f"finished creating {pipeline_gibbs_syn}")
                pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                         pipeline_gt_config)
                print(f"finished creating {pipeline_gt_syn}")

                print("matrices compared l1 norm")
                results[0, j, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat,
                                                       pipeline_pome_syn.transmat)
                results[1, j, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat,
                                                       pipeline_gibbs_syn.transmat)
                x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

        for j in range(n_run):
            axes1[l].plot(x, results[0, j], marker='.', label=f"run #{j + 1}", linestyle=":", )
            axes1[l].set_title(transmat_mode)
            axes1[l].set_ylabel("L1")

            axes2[l].plot(x, results[1, j], marker='.', label=f"run #{j + 1}", linestyle=":", )
            axes2[l].set_title(transmat_mode)
            axes2[l].set_ylabel("L1")
        axes1[l].legend()
        axes2[l].legend()

    fig1.suptitle("Pome " + omitter)
    fig1.supxlabel("temporal information")
    fig1.savefig("Pome " + omitter + " non avg")
    plt.savefig("")
    fig2.suptitle("Gibbs " + omitter)
    fig2.supxlabel("temporal information")
    fig2.savefig("Gibbs " + omitter + " non avg")
    if show:
        plt.show()


def compare_pipelines_transmat_mode_with_vs_temp_info_var_norm(omitter, n_run, powers,
                                                               transmat_modes=None,
                                                               show=True):
    fig1, axes1 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))
    fig2, axes2 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))
    for l, transmat_mode in enumerate(transmat_modes):
        results = np.empty((2, n_run, len(powers[l])))
        reader = "My Synthetic"
        for j in range(n_run):
            reader_config: my_synthetic_reader_config
            reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
                reader, omitter,
                "Pomegranate - Synthetic" + str(j + 1))
            _, _, pipeline_gibbs_config = create_config(reader, omitter,
                                                        "Gibbs Sampler" + str(j + 1))
            _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                       "Ground Truth" + str(j + 1))
            reader_config.transmat_mode = transmat_mode

            x = []
            for k, power in enumerate(powers[l]):
                reader_config.set_temporal = True
                reader_config.matrix_power = power

                pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                           pipeline_pome_config)
                print(f"finished creating {pipeline_pome_syn}")
                pipeline_gibbs_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                            pipeline_gibbs_config)
                print(f"finished creating {pipeline_gibbs_syn}")
                pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                         pipeline_gt_config)
                print(f"finished creating {pipeline_gt_syn}")

                print("matrices compared l1 norm")
                pome_stochastic_value = compare_mat_l1_norm(pipeline_gt_syn.transmat_list[0],
                                                            pipeline_pome_syn.transmat_list[0])
                gibbs_stochastic_value = compare_mat_l1_norm(pipeline_gt_syn.transmat_list[0],
                                                             pipeline_gibbs_syn.transmat_list[0])
                results[0, j, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat,
                                                       pipeline_pome_syn.transmat) / pome_stochastic_value
                results[1, j, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat,
                                                       pipeline_gibbs_syn.transmat) / gibbs_stochastic_value
                x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

        for j in range(n_run):
            axes1[l].plot(x, results[0, j], marker='.', label=f"run #{j + 1}", linestyle=":", )
            axes1[l].set_title(transmat_mode)
            axes1[l].set_ylabel("L1")

            axes2[l].plot(x, results[1, j], marker='.', label=f"run #{j + 1}", linestyle=":", )
            axes2[l].set_title(transmat_mode)
            axes2[l].set_ylabel("L1")
        axes1[l].legend()
        axes2[l].legend()

    fig1.suptitle("Pome " + omitter)
    fig1.supxlabel("temporal information")
    fig1.savefig("Pome " + omitter + " non avg" + "Norm")
    plt.savefig("")
    fig2.suptitle("Gibbs " + omitter)
    fig2.supxlabel("temporal information")
    fig2.savefig("Gibbs " + omitter + " non avg" + "Norm")
    if show:
        plt.show()


def compare_pipelines_transmat_mode_with_vs_temp_info_avg(omitter, n_run, powers,
                                                          transmat_modes=None,
                                                          show=True):
    fig1, axes1 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))
    fig2, axes2 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))
    for l, transmat_mode in enumerate(transmat_modes):
        results = np.empty((2, n_run, len(powers[l])))
        reader = "My Synthetic"
        for j in range(n_run):
            reader_config: my_synthetic_reader_config
            reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
                reader, omitter,
                "Pomegranate - Synthetic" + str(j + 1))
            _, _, pipeline_gibbs_config = create_config(reader, omitter,
                                                        "Gibbs Sampler" + str(j + 1))
            _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                       "Ground Truth" + str(j + 1))
            reader_config.transmat_mode = transmat_mode

            x = []
            for k, power in enumerate(powers[l]):
                reader_config.set_temporal = True
                reader_config.matrix_power = power

                pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                           pipeline_pome_config)
                print(f"finished creating {pipeline_pome_syn}")
                pipeline_gibbs_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                            pipeline_gibbs_config)
                print(f"finished creating {pipeline_gibbs_syn}")
                pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                         pipeline_gt_config)
                print(f"finished creating {pipeline_gt_syn}")

                print("matrices compared l1 norm")
                results[0, j, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat,
                                                       pipeline_pome_syn.transmat)
                results[1, j, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat,
                                                       pipeline_gibbs_syn.transmat)
                x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

        results = np.average(results, axis=1)
        axes1[l].plot(x, results[0], marker='.', label=f"run #{j + 1}", linestyle=":", )
        axes1[l].set_title(transmat_mode)
        axes1[l].set_ylabel("L1")

        axes2[l].plot(x, results[1], marker='.', label=f"run #{j + 1}", linestyle=":", )
        axes2[l].set_title(transmat_mode)
        axes2[l].set_ylabel("L1")
        axes1[l].legend()
        axes2[l].legend()

    fig1.suptitle("Pome " + omitter)
    fig1.supxlabel("temporal information")
    fig1.savefig("Pome " + omitter + " avg")
    plt.savefig("")
    fig2.suptitle("Gibbs " + omitter)
    fig2.supxlabel("temporal information")
    fig2.savefig("Gibbs " + omitter + " avg")
    if show:
        plt.show()


def compare_pipelines_transmat_mode_with_vs_temp_info_avg_norm(omitter, n_run, powers,
                                                               transmat_modes=None,
                                                               show=True):
    fig1, axes1 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))
    fig2, axes2 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))
    for l, transmat_mode in enumerate(transmat_modes):
        results = np.empty((2, n_run, len(powers[l])))
        reader = "My Synthetic"
        for j in range(n_run):
            reader_config: my_synthetic_reader_config
            reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
                reader, omitter,
                "Pomegranate - Synthetic" + str(j + 1))
            _, _, pipeline_gibbs_config = create_config(reader, omitter,
                                                        "Gibbs Sampler" + str(j + 1))
            _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                       "Ground Truth" + str(j + 1))
            reader_config.transmat_mode = transmat_mode

            x = []
            for k, power in enumerate(powers[l]):
                reader_config.set_temporal = True
                reader_config.matrix_power = power

                pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                           pipeline_pome_config)
                print(f"finished creating {pipeline_pome_syn}")
                pipeline_gibbs_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                            pipeline_gibbs_config)
                print(f"finished creating {pipeline_gibbs_syn}")
                pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                         pipeline_gt_config)
                print(f"finished creating {pipeline_gt_syn}")

                print("matrices compared l1 norm")
                pome_stochastic_value = compare_mat_l1_norm(pipeline_gt_syn.transmat_list[0],
                                                            pipeline_pome_syn.transmat_list[0])
                gibbs_stochastic_value = compare_mat_l1_norm(pipeline_gt_syn.transmat_list[0],
                                                             pipeline_gibbs_syn.transmat_list[0])
                results[0, j, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat,
                                                       pipeline_pome_syn.transmat) / pome_stochastic_value
                results[1, j, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat,
                                                       pipeline_gibbs_syn.transmat) / gibbs_stochastic_value
                x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

        results = np.average(results, axis=1)
        axes1[l].plot(x, results[0], marker='.', linestyle=":", )
        axes1[l].set_title(transmat_mode)
        axes1[l].set_ylabel("L1")

        axes2[l].plot(x, results[1], marker='.', linestyle=":", )
        axes2[l].set_title(transmat_mode)
        axes2[l].set_ylabel("L1")



    fig1.suptitle("Pome " + omitter)
    fig1.supxlabel("temporal information")
    fig1.savefig("Pome " + omitter + " avg" + "Norm")
    fig2.suptitle("Gibbs " + omitter)
    fig2.supxlabel("temporal information")
    fig2.savefig("Gibbs " + omitter + " avg" + "Norm")
    if show:
        plt.show()


def compare_pipelines_for_different_prob_transmat_modes_with_temp_info_vs_prob_norm_pome(omitter, n_run, powers,
                                                                                         transmat_modes=None,
                                                                                         n_probabilities=None,
                                                                                         probabilities=None,
                                                                                         show=True):
    fig1, axes1 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))
    for l, transmat_mode in enumerate(transmat_modes):
        if n_probabilities is None:
            n_probabilities = len(probabilities)
        if probabilities is None:
            probabilities = np.linspace(0.1, 1, n_probabilities)
        results = np.empty((2, n_run, n_probabilities, len(powers[l])))
        reader = "My Synthetic"
        for j in range(n_run):
            omitter_bernoulli_config: bernoulli_omitter_config
            reader_config: my_synthetic_reader_config
            reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
                reader, omitter,
                "Pomegranate - Synthetic" + str(j + 1))
            _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                       "Ground Truth" + str(j + 1))
            reader_config.transmat_mode = transmat_mode

            for i, p in enumerate(probabilities):
                omitter_bernoulli_config.prob_of_observation = p  # change the prob
                x = []
                for k, power in enumerate(powers[l]):
                    reader_config.set_temporal = True
                    reader_config.matrix_power = power

                    pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                               pipeline_pome_config)
                    print(f"finished creating {pipeline_pome_syn}")
                    pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                             pipeline_gt_config)
                    print(f"finished creating {pipeline_gt_syn}")

                    print("matrices compared l1 norm")
                    pome_stochastic_value = compare_mat_l1_norm(pipeline_gt_syn.transmat_list[0],
                                                                pipeline_pome_syn.transmat_list[0])
                    results[0, j, i, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat,
                                                              pipeline_pome_syn.transmat) / pome_stochastic_value
                    x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

        results = np.average(results, axis=1)
        cmap = cm.get_cmap('viridis')
        for k, power in enumerate(powers[l]):
            color = cmap(x[k] / max(x))  # Normalize x[k] to [0, 1]
            axes1[l].plot(probabilities, results[0, :, k], marker='.', label=f"temp info = {x[k]:.2f}", linestyle=":",
                          color=color)
            axes1[l].set_title(transmat_mode)
            axes1[l].set_ylabel("L1")
        axes1[l].legend()

    fig1.suptitle("Pome " + omitter)
    fig1.supxlabel("Probability")
    fig1.savefig("Pome" + omitter + "Norm")

    if show:
        plt.show()


def compare_pipelines_for_different_prob_transmat_modes_with_temp_info_vs_iter_pome(omitter, n_run, powers, n_iter,
                                                                                    transmat_mode=None,
                                                                                    n_probabilities=None,
                                                                                    probabilities=None,
                                                                                    show=True):
    if n_probabilities is None:
        n_probabilities = len(probabilities)
    if probabilities is None:
        probabilities = np.linspace(0.1, 1, n_probabilities)
    fig1, axes1 = plt.subplots(n_probabilities, 1, sharex=True, sharey=True, figsize=(height, width))

    results = np.empty((2, n_run, n_probabilities, len(powers), n_iter + 1))
    reader = "My Synthetic"
    for j in range(n_run):
        omitter_bernoulli_config: bernoulli_omitter_config
        reader_config: my_synthetic_reader_config
        reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
            reader, omitter,
            "Pomegranate - Synthetic" + str(j + 1))
        _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                   "Ground Truth" + str(j + 1))
        reader_config.transmat_mode = transmat_mode

        for i, p in enumerate(probabilities):
            omitter_bernoulli_config.prob_of_observation = p  # change the prob
            x = []
            for k, power in enumerate(powers):
                reader_config.set_temporal = True
                reader_config.matrix_power = power

                pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                           pipeline_pome_config)
                print(f"finished creating {pipeline_pome_syn}")
                pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                         pipeline_gt_config)
                print(f"finished creating {pipeline_gt_syn}")

                print("matrices compared l1 norm")
                results[0, j, i, k] = compare_mat_l1_norm_for_list(pipeline_gt_syn.transmat_list,
                                                                   pipeline_pome_syn.transmat_list)
                results[0, j, i, k] = results[0, j, i, k] / results[0, j, i, k, 0]
                x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

    results = np.average(results, axis=1)
    cmap = cm.get_cmap('viridis')

    for i, probability in enumerate(probabilities):
        for k, power in enumerate(powers):
            color = cmap(x[k] / max(x))  # Normalize x[k] to [0, 1]
            axes1[i].plot(np.arange(n_iter + 1), results[0, i, k], marker='.', label=f"temp info = {x[k]:.2f}",
                          linestyle=":",
                          color=color)
            axes1[i].set_title(f"p = {probability:.2f}")
            axes1[i].set_ylabel("L1")
        axes1[i].legend()

    fig1.suptitle("Pome " + omitter + " " + transmat_mode)
    fig1.supxlabel("# Iter")
    fig1.savefig("Pome " + omitter + " " + transmat_mode + " Iter")

    if show:
        plt.show()

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

import Config.Config
from Experiments.utils import create_config_dataclass_objects as create_config
from Experiments.utils import create_and_fit_pipeline as create_pipeline
from Experiments.utils import load_or_initialize_pipeline as create_pipeline_from_configs
from Pipelines.pipeline import pipeline
from Experiments.evalUtils import *
from Config.Config import *

height = 10
width = 7


def simple_valid_pipeline(pipeline_to_check: pipeline):
    print(pipeline_to_check.transmat)
    print(pipeline_to_check.startprob)
    print(len(pipeline_to_check.transmat_list))
    print(len(pipeline_to_check.startprob_list))


def simple_validate_pipelines():
    ## TODO: Create default for readers, omitters, models
    ## TODO: Create dataclasses for configurations, as seen in playground.
    # Create dict of attributes from the dataclass, and fill it up using json and default.
    # Then create the dataclass and pass to the object
    # and then the model etc will hold a config obj

    # region pome, pass all, simple valid
    """pipeline_pome_stocks: pipeline = create_pipeline("Stocks", "Pass All", "Pomegranate - Stocks")
    print(f"finished creating {pipeline_pome_stocks.__str__()}")
    pipeline_pome_syn: pipeline = create_pipeline("Synthetic", "Pass All", "Pomegranate - Synthetic")
    print(f"finished creating {pipeline_pome_syn.__str__()}")
    pipeline_pome_brown: pipeline = create_pipeline("Brown Corpus", "Pass All", "Pomegranate - Brown")
    print(f"finished creating {pipeline_pome_brown.__str__()}")

    simple_valid_pipeline(pipeline_pome_stocks)
    simple_valid_pipeline(pipeline_pome_syn)
    simple_valid_pipeline(pipeline_pome_brown)"""
    # endregion

    # region pome, bernoulli, simple valid

    """pipeline_pome_stocks: pipeline = create_pipeline("Stocks", "Bernoulli", "Pomegranate - Stocks")
    print(f"finished creating {pipeline_pome_stocks.__str__()}")
    pipeline_pome_syn: pipeline = create_pipeline("Synthetic", "Bernoulli", "Pomegranate - Synthetic")
    print(f"finished creating {pipeline_pome_syn.__str__()}")
    pipeline_pome_brown: pipeline = create_pipeline("Brown Corpus", "Bernoulli", "Pomegranate - Brown")
    print(f"finished creating {pipeline_pome_brown.__str__()}")

    simple_valid_pipeline(pipeline_pome_stocks)
    simple_valid_pipeline(pipeline_pome_syn)
    simple_valid_pipeline(pipeline_pome_brown)"""
    # endregion

    # region gibbs, pass all, simple valid
    pipeline_gibbs_syn: pipeline = create_pipeline("Synthetic", "Pass All", "Gibbs Sampler")
    print(f"finished creating {pipeline_gibbs_syn.__str__()}")

    simple_valid_pipeline(pipeline_gibbs_syn)
    print(pipeline_gibbs_syn.startprob_list)

    # endregion
    # region gibbs, bernoulli, simple valid
    pipeline_gibbs_syn: pipeline = create_pipeline("Synthetic", "Bernoulli", "Gibbs Sampler")
    print(f"finished creating {pipeline_gibbs_syn.__str__()}")

    simple_valid_pipeline(pipeline_gibbs_syn)
    print(pipeline_gibbs_syn.startprob_list)

    # endregion


def compare_synthetic_transmat():
    pipeline_pome_syn: pipeline = create_pipeline("Synthetic", "Pass All", "Pomegranate - Synthetic")
    print(f"finished creating {pipeline_pome_syn}")
    pipeline_hmm_syn: pipeline = create_pipeline("Synthetic", "Pass All", "Hmmlearn")
    print(f"finished creating {pipeline_hmm_syn}")
    pipeline_gibbs_syn: pipeline = create_pipeline("Synthetic", "Pass All", "Gibbs Sampler")
    print(f"finished creating {pipeline_gibbs_syn}")
    pipeline_gt_syn: pipeline = create_pipeline("Synthetic", "Pass All", "Ground Truth")
    print(f"finished creating {pipeline_gt_syn}")

    print(pipeline_pome_syn.reader.transmat)
    print(pipeline_hmm_syn.reader.transmat)
    print(pipeline_gibbs_syn.reader.transmat)
    print(pipeline_gt_syn.reader.transmat)

    print("matrices")
    print(pipeline_pome_syn.transmat)
    print(pipeline_hmm_syn.transmat)
    print(pipeline_gibbs_syn.transmat)
    print(pipeline_gt_syn.transmat)

    print("matrices norm")
    print(np.sum(pipeline_pome_syn.transmat, axis=1))
    print(np.sum(pipeline_hmm_syn.transmat, axis=1))
    print(np.sum(pipeline_gibbs_syn.transmat, axis=1))
    print(np.sum(pipeline_gt_syn.transmat, axis=1))

    print("matrices compared l1 norm")
    compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_pome_syn.transmat)
    compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_hmm_syn.transmat)
    compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_gibbs_syn.transmat)

    print("means")
    print(pipeline_pome_syn.means)
    print(pipeline_hmm_syn.means)
    print(pipeline_gt_syn.means)


def compare_hmm_synthetic_transmat():
    pipeline_pome_syn: pipeline = create_pipeline("HMM - Synthetic", "Pass All", "Pomegranate - Synthetic")
    print(f"finished creating {pipeline_pome_syn}")
    pipeline_hmm_syn: pipeline = create_pipeline("HMM - Synthetic", "Pass All", "Hmmlearn")
    print(f"finished creating {pipeline_hmm_syn}")
    pipeline_gibbs_syn: pipeline = create_pipeline("HMM - Synthetic", "Pass All", "Gibbs Sampler")
    print(f"finished creating {pipeline_gibbs_syn}")
    pipeline_gt_syn: pipeline = create_pipeline("HMM - Synthetic", "Pass All", "Ground Truth")
    print(f"finished creating {pipeline_gt_syn}")

    print("matrices")
    print(pipeline_pome_syn.transmat)
    print(pipeline_hmm_syn.transmat)
    print(pipeline_gibbs_syn.transmat)
    print(pipeline_gt_syn.transmat)
    print()

    print("matrices norm")
    print(np.sum(pipeline_pome_syn.transmat, axis=1))
    print(np.sum(pipeline_hmm_syn.transmat, axis=1))
    print(np.sum(pipeline_gibbs_syn.transmat, axis=1))
    print(np.sum(pipeline_gt_syn.transmat, axis=1))

    print("matrices compared l1 norm")
    compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_pome_syn.transmat)
    compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_hmm_syn.transmat)
    compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_gibbs_syn.transmat)

    print("means")
    print(pipeline_pome_syn.means)
    print(pipeline_hmm_syn.means)
    print(pipeline_gt_syn.means)


def compare_pipelines_for_different_sigmas():
    readers = ["Synthetic Hard"]
    num_of_run = 5
    results = np.empty((2, 1, num_of_run))
    fig, ax = plt.subplots(num_of_run, 1, sharex=True, sharey=True, figsize=(10, 6))
    for j in range(1, num_of_run + 1):
        for i, reader in enumerate(readers):
            pipeline_pome_syn: pipeline = create_pipeline(reader, "Pass All", "Pomegranate - Synthetic" + str(j))
            print(f"finished creating {pipeline_pome_syn}")
            """pipeline_hmm_syn: pipeline = create_pipeline(reader, "Pass All", "Hmmlearn" + str(j))
            print(f"finished creating {pipeline_hmm_syn}")"""
            pipeline_gibbs_syn: pipeline = create_pipeline(reader, "Pass All", "Gibbs Sampler" + str(j))
            print(f"finished creating {pipeline_gibbs_syn}")
            pipeline_gt_syn: pipeline = create_pipeline(reader, "Pass All", "Ground Truth" + str(j))
            print(f"finished creating {pipeline_gt_syn}")

            print("matrices compared l1 norm")
            pome_perm = find_optimal_permutation(pipeline_pome_syn.means, pipeline_gt_syn.means)
            # hmm_perm = find_optimal_permutation(pipeline_hmm_syn.means, pipeline_gt_syn.means)
            gibbs_perm = find_optimal_permutation(pipeline_gibbs_syn.means, pipeline_gt_syn.means)

            results[0, i, j - 1] = find_mat_diff(pipeline_gt_syn.transmat,
                                                 reorient_matrix(pipeline_pome_syn.transmat, pome_perm)) / \
                                   pipeline_gt_syn.transmat.shape[0]
            """results[1, i, j - 1] = find_mat_diff(pipeline_gt_syn.transmat,
                                                 reorient_matrix(pipeline_hmm_syn.transmat, hmm_perm)) / \
                                   pipeline_gt_syn.transmat.shape[0]"""
            results[1, i, j - 1] = find_mat_diff(pipeline_gt_syn.transmat, pipeline_gibbs_syn.transmat) / \
                                   pipeline_gt_syn.transmat.shape[
                                       0]  # we don't change the orientation of the gibbs matrix

        x = [0.01]
        ax[j - 1].plot(x, results[0, :, j - 1], marker='D', label="pome", linestyle=":")
        ax[j - 1].plot(x, results[1, :, j - 1], marker='D', label="gibbs", linestyle=":")
        ax[j - 1].set_title(f"Run  #{j}")
        ax[j - 1].set_xscale('log')
        ax[j - 1].yaxis.set_major_locator(plt.MaxNLocator(5))

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower left')
    fig.subplots_adjust(hspace=0.5)
    fig.supxlabel("Sigma")
    fig.supylabel("L1 normalized")
    plt.show()
    plt.figure(10)
    x = [0.01]
    results = np.average(results, axis=2)
    plt.plot(x, results[0, :], marker='o', label="pome", linestyle=":")
    plt.plot(x, results[1, :], marker='o', label="gibbs", linestyle=":")
    plt.ylabel("L1 normalized")
    plt.xlabel("Sigma")
    plt.xscale("log")
    plt.title("Averaged Iterations")
    plt.legend()
    print(pipeline_gt_syn.reader.dataset['lengths'])


def compare_pipelines_vs_iter_pass_all(n_components, n_run, n_iter, show=True):
    reader = "My Synthetic"
    results = np.empty((2, n_run, n_iter + 1))
    means_results = np.empty((2, n_run, n_iter + 1, n_components))
    for j in range(n_run):
        pipeline_pome_syn: pipeline = create_pipeline(reader, "Pass All", "Pomegranate - Synthetic" + str(j + 1))
        print(f"finished creating {pipeline_pome_syn}")
        pipeline_gibbs_syn: pipeline = create_pipeline(reader, "Pass All", "Gibbs Sampler" + str(j + 1))
        print(f"finished creating {pipeline_gibbs_syn}")
        pipeline_gt_syn: pipeline = create_pipeline(reader, "Pass All", "Ground Truth" + str(j + 1))
        print(f"finished creating {pipeline_gt_syn}")

        print("matrices compared l1 norm")

        results[0, j, :] = np.array(
            compare_mat_l1_norm_for_list(pipeline_gt_syn.transmat_list, pipeline_pome_syn.transmat_list))
        means_results[0, j, :, :] = np.array(pipeline_pome_syn.means_list)
        results[1, j, :] = np.array(
            compare_mat_l1_norm_for_list(pipeline_gt_syn.transmat_list, pipeline_gibbs_syn.transmat_list))
        means_results[1, j, :, :] = np.array(pipeline_gibbs_syn.means_list)
        x = np.arange(n_iter + 1)
        plt.figure(1)
        plt.plot(x, results[0, j, :], marker='.', label=f"run #{j}", linestyle=":")
        plt.figure(2)
        plt.plot(x, results[1, j, :], marker='.', label=f"run #{j}", linestyle=":")

        plt.figure(3 + 2 * j)
        plt.plot(x, means_results[0, j, :, :], marker='.', label=f"run #{j}", linestyle="")
        plt.title(f"Pome run #{j}")
        plt.axhline(y=1, color='r', linestyle=':')
        plt.axhline(y=2, color='r', linestyle=':')
        plt.axhline(y=3, color='r', linestyle=':')
        plt.figure(4 + 2 * j)
        plt.plot(x, means_results[1, j, :, :], marker='.', label=f"run #{j}", linestyle="")
        plt.title(f"Gibbs run #{j}")
        print(np.sum(pipeline_gt_syn.reader.dataset['lengths']))

    plt.figure(1)
    plt.title("Pome")
    plt.xlabel("iter #")
    plt.ylabel("L1 Norm")
    plt.legend()
    plt.figure(2)
    plt.title("Gibbs")
    plt.xlabel("iter #")
    plt.ylabel("L1 Norm")
    plt.legend()

    if show:
        plt.show()


def compare_pipelines_for_different_prob(omitter, n_run, n_probabilities=None, probabilities=None, show=True):
    if n_probabilities is None:
        n_probabilities = len(probabilities)
    if probabilities is None:
        probabilities = np.linspace(0.1, 1, n_probabilities)
    results = np.empty((2, n_run, n_probabilities))
    reader = "My Synthetic"
    for j in range(n_run):
        for i, p in enumerate(probabilities):
            omitter_bernoulli_config: bernoulli_omitter_config
            reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
                reader, omitter,
                "Pomegranate - Synthetic" + str(j + 1))
            _, _, pipeline_gibbs_config = create_config(reader, "Bernoulli",
                                                        "Gibbs Sampler" + str(j + 1))
            _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                       "Ground Truth" + str(j + 1))

            omitter_bernoulli_config.prob_of_observation = p  # change the prob
            pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                       pipeline_pome_config)
            print(f"finished creating {pipeline_pome_syn}")
            pipeline_gibbs_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                        pipeline_gibbs_config)
            print(f"finished creating {pipeline_gibbs_syn}")
            pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                     pipeline_gt_config)
            print(f"finished creating {pipeline_gt_syn}")

            print("matrices compared l1 norm")
            results[0, j, i] = np.array(
                compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_pome_syn.transmat))
            results[1, j, i] = np.array(
                compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_gibbs_syn.transmat))
        plt.figure(1)
        plt.plot(probabilities, results[0, j, :], marker='.', label=f"run #{j}", linestyle=":")
        plt.figure(2)
        plt.plot(probabilities, results[1, j, :], marker='.', label=f"run #{j}", linestyle=":")

    plt.figure(1)
    plt.title("Pome " + omitter)
    plt.xlabel("probability of seeing emission")
    plt.ylabel("L1 Norm")
    plt.legend()
    plt.figure(2)
    plt.title("Gibbs " + omitter)
    plt.xlabel("probability of seeing emission")
    plt.ylabel("L1 Norm")
    plt.legend()

    if show:
        plt.show()


def compare_pipelines_for_different_prob_pome(omitter, n_run, n_probabilities=None, probabilities=None, show=True):
    if n_probabilities is None:
        n_probabilities = len(probabilities)
    if probabilities is None:
        probabilities = np.linspace(0.1, 1, n_probabilities)
    results = np.empty((n_run, n_probabilities))
    reader = "My Synthetic"
    for j in range(n_run):
        for i, p in enumerate(probabilities):
            omitter_bernoulli_config: bernoulli_omitter_config
            reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
                reader, omitter,
                "Pomegranate - Synthetic" + str(j + 1))
            _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                       "Ground Truth" + str(j + 1))

            omitter_bernoulli_config.prob_of_observation = p  # change the prob
            pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                       pipeline_pome_config)
            print(f"finished creating {pipeline_pome_syn}")
            pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                     pipeline_gt_config)
            print(f"finished creating {pipeline_gt_syn}")

            print("matrices compared l1 norm")
            results[j, i] = np.array(
                compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_pome_syn.transmat))
        plt.figure(1)
        plt.plot(probabilities, results[j, :], marker='.', label=f"run #{j}", linestyle=":")

    plt.figure(1)
    plt.title("Pome " + omitter)
    plt.xlabel("probability of seeing emission")
    plt.ylabel("L1 Norm")
    plt.legend()

    if show:
        plt.show()


def compare_pipelines_vs_iter_pass_all_different_sample_len(n_run, n_iter, n_samples_arr, sentence_length_arr,
                                                            n_components=10, show=True):
    n_sample_len = len(n_samples_arr)
    if len(sentence_length_arr) != n_sample_len:
        raise ValueError(
            f"Sentence Length Array:{sentence_length_arr} does not have the same number of elements as the Number of Samples Array: {n_samples_arr} ")
    reader = "My Synthetic"
    results = np.empty((2, n_sample_len, n_run, n_iter + 1))
    for i, (n_samples, sentence_length) in enumerate(zip(n_samples_arr, sentence_length_arr)):
        for j in range(n_run):
            reader_config: my_synthetic_reader_config
            reader_config, omitter_config, pipeline_pome_config = create_config(reader, "Pass All",
                                                                                "Pomegranate - Synthetic" + str(j + 1))
            _, _, pipeline_gibbs_config = create_config(reader, "Pass All",
                                                        "Gibbs Sampler" + str(j + 1))
            _, _, pipeline_gt_config = create_config(reader, "Pass All",
                                                     "Ground Truth" + str(j + 1))

            reader_config.n_samples = n_samples
            reader_config.sentence_length = sentence_length
            pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_config,
                                                                       pipeline_pome_config)
            print(f"finished creating {pipeline_pome_syn}")
            pipeline_gibbs_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_config,
                                                                        pipeline_gibbs_config)
            print(f"finished creating {pipeline_gibbs_syn}")
            pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_config,
                                                                     pipeline_gt_config)
            print(f"finished creating {pipeline_gt_syn}")
            print("matrices compared l1 norm")

            results[0, i, j] = np.array(
                compare_mat_l1_norm_for_list(pipeline_gt_syn.transmat_list, pipeline_pome_syn.transmat_list))
            results[1, i, j] = np.array(
                compare_mat_l1_norm_for_list(pipeline_gt_syn.transmat_list, pipeline_gibbs_syn.transmat_list))
            x = np.arange(n_iter + 1)
            plt.figure(2 * i)
            plt.plot(x, results[0, i, j], marker='.', label=f"run #{j}", linestyle=":")
            plt.figure(2 * i + 1)
            plt.plot(x, results[1, i, j], marker='.', label=f"run #{j}", linestyle=":")
        plt.figure(2 * i)
        plt.title(f"Pome - Sentence Length:{sentence_length}, Number of Samples: {n_samples}")
        plt.xlabel("iter #")
        plt.ylabel("L1 Norm")
        plt.legend()
        plt.figure(2 * i + 1)
        plt.title(f"Gibbs - Sentence Length:{sentence_length}, Number of Samples: {n_samples}")
        plt.plot(x, results[1, i, j], marker='.', label=f"run #{j}", linestyle=":")
        plt.xlabel("iter #")
        plt.ylabel("L1 Norm")
        plt.legend()

    if show:
        plt.show()


def compare_pipelines_for_different_prob_vs_iter(omitter, n_run, n_iter, n_probabilities=None,
                                                 probabilities=None, show=True):
    if n_probabilities is None:
        n_probabilities = len(probabilities)
    if probabilities is None:
        probabilities = np.linspace(0.1, 1, n_probabilities)
    results = np.empty((2, n_run, n_probabilities, n_iter + 1))

    fig1, axes1 = plt.subplots(n_probabilities, 1, sharex=True, sharey=True, figsize=(10, 6))

    fig2, axes2 = plt.subplots(n_probabilities, 1, sharex=True, sharey=True, figsize=(10, 6))
    reader = "My Synthetic"
    for j in range(n_run):
        for i, p in enumerate(probabilities):
            omitter_bernoulli_config: bernoulli_omitter_config
            reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
                reader, omitter,
                "Pomegranate - Synthetic" + str(j + 1))
            _, _, pipeline_gibbs_config = create_config(reader, "Bernoulli",
                                                        "Gibbs Sampler" + str(j + 1))
            _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                       "Ground Truth" + str(j + 1))

            omitter_bernoulli_config.prob_of_observation = p  # change the prob
            pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                       pipeline_pome_config)
            print(f"finished creating {pipeline_pome_syn}")
            pipeline_gibbs_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                        pipeline_gibbs_config)
            print(f"finished creating {pipeline_gibbs_syn}")
            pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                     pipeline_gt_config)
            print(f"finished creating {pipeline_gt_syn}")

            print("matrices compared l1 norm")
            results[0, j, i] = np.array(
                compare_mat_l1_norm_for_list(pipeline_gt_syn.transmat_list, pipeline_pome_syn.transmat_list))
            results[1, j, i] = np.array(
                compare_mat_l1_norm_for_list(pipeline_gt_syn.transmat_list, pipeline_gibbs_syn.transmat_list))
            x = np.arange(n_iter + 1)

            axes1[i].plot(x, results[0, j, i], marker='.', label=f"run #{j}", linestyle=":")
            axes1[i].set_title(f"p={p}")

            axes2[i].plot(x, results[1, j, i], marker='.', label=f"run #{j}", linestyle=":")
            axes2[i].set_title(f"p={p}")

    handles, labels = axes1[0].get_legend_handles_labels()
    fig1.legend(handles, labels, loc='lower left')
    fig1.suptitle("Pome " + omitter)
    fig1.subplots_adjust(hspace=0.6)
    fig1.supxlabel("Iter #")
    handles, labels = axes2[0].get_legend_handles_labels()
    fig2.legend(handles, labels, loc='lower left')
    fig2.suptitle("Gibbs " + omitter)
    fig2.subplots_adjust(hspace=0.6)
    fig2.supxlabel("Iter #")
    if show:
        plt.show()


def compare_pipelines_for_different_prob_vs_iter(omitter, n_run, n_iter, n_probabilities=None,
                                                 probabilities=None, show=True):
    if n_probabilities is None:
        n_probabilities = len(probabilities)
    if probabilities is None:
        probabilities = np.linspace(0.1, 1, n_probabilities)
    results = np.empty((n_run, n_probabilities, n_iter + 1))

    fig1, axes1 = plt.subplots(n_probabilities, 1, sharex=True, sharey=True, figsize=(10, 6))

    reader = "My Synthetic"
    for j in range(n_run):
        for i, p in enumerate(probabilities):
            omitter_bernoulli_config: bernoulli_omitter_config
            reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
                reader, omitter,
                "Pomegranate - Synthetic" + str(j + 1))
            _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                       "Ground Truth" + str(j + 1))

            omitter_bernoulli_config.prob_of_observation = p  # change the prob
            pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                       pipeline_pome_config)
            print(f"finished creating {pipeline_pome_syn}")
            pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                     pipeline_gt_config)
            print(f"finished creating {pipeline_gt_syn}")

            print("matrices compared l1 norm")
            results[j, i] = np.array(
                compare_mat_l1_norm_for_list(pipeline_gt_syn.transmat_list, pipeline_pome_syn.transmat_list))
            x = np.arange(n_iter + 1)

            axes1[i].plot(x, results[j, i], marker='.', label=f"run #{j}", linestyle=":")
            axes1[i].set_title(f"p={p}")

    handles, labels = axes1[0].get_legend_handles_labels()
    fig1.legend(handles, labels, loc='lower left')
    fig1.suptitle("Pome " + omitter)
    fig1.subplots_adjust(hspace=0.6)
    fig1.supxlabel("Iter #")
    if show:
        plt.show()


def compare_pipelines_for_different_prob_vs_temporal_info(omitter, n_run, powers, n_probabilities=None,
                                                          probabilities=None, show=True):
    if n_probabilities is None:
        n_probabilities = len(probabilities)
    if probabilities is None:
        probabilities = np.linspace(0.1, 1, n_probabilities)
    results = np.empty((2, n_run, n_probabilities, len(powers)))

    fig1, axes1 = plt.subplots(n_probabilities, 1, sharex=True, sharey=True, figsize=(10, 6))

    fig2, axes2 = plt.subplots(n_probabilities, 1, sharex=True, sharey=True, figsize=(10, 6))
    reader = "My Synthetic"
    for j in range(n_run):
        omitter_bernoulli_config: bernoulli_omitter_config
        reader_config: my_synthetic_reader_config
        reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
            reader, omitter,
            "Pomegranate - Synthetic" + str(j + 1))
        _, _, pipeline_gibbs_config = create_config(reader, "Bernoulli",
                                                    "Gibbs Sampler" + str(j + 1))
        _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                   "Ground Truth" + str(j + 1))

        for i, p in enumerate(probabilities):
            omitter_bernoulli_config.prob_of_observation = p  # change the prob
            x = []
            for k, power in enumerate(powers):
                reader_config.set_temporal = True
                reader_config.matrix_power = power

                pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                           pipeline_pome_config)
                print(f"finished creating {pipeline_pome_syn}")
                pipeline_gibbs_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                            pipeline_gibbs_config)
                print(f"finished creating {pipeline_gibbs_syn}")
                pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                         pipeline_gt_config)
                print(f"finished creating {pipeline_gt_syn}")

                print("matrices compared l1 norm")
                results[0, j, i, k - 1] = compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_pome_syn.transmat)
                results[1, j, i, k - 1] = compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_gibbs_syn.transmat)
                x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

            axes1[i].plot(x, results[0, j, i, :], marker='.', label=f"run #{j}", linestyle=":")
            axes1[i].set_title(f"p={p}")

            axes2[i].plot(x, results[1, j, i, :], marker='.', label=f"run #{j}", linestyle=":")
            axes2[i].set_title(f"p={p}")

    handles, labels = axes1[0].get_legend_handles_labels()
    fig1.legend(handles, labels, loc='lower left')
    fig1.suptitle("Pome " + omitter)
    fig1.subplots_adjust(hspace=0.6)
    fig1.supxlabel("Temporal Info Ratio")
    handles, labels = axes2[0].get_legend_handles_labels()
    fig2.legend(handles, labels, loc='lower left')
    fig2.suptitle("Gibbs " + omitter)
    fig2.subplots_adjust(hspace=0.6)
    fig2.supxlabel("Temporal Info Ratio")
    if show:
        plt.show()


def compare_pipelines_for_different_prob_transmat_mode_vs_temporal_info_pome(omitter, n_run, powers, transmat_mode=None,
                                                                             n_probabilities=None, probabilities=None,
                                                                             show=True):
    if n_probabilities is None:
        n_probabilities = len(probabilities)
    if probabilities is None:
        probabilities = np.linspace(0.1, 1, n_probabilities)
    results = np.empty((n_run, n_probabilities, len(powers)))

    fig1, axes1 = plt.subplots(n_probabilities, 1, sharex=True, sharey=True, figsize=(10, 6))

    reader = "My Synthetic"
    for j in range(n_run):
        omitter_bernoulli_config: bernoulli_omitter_config
        reader_config: my_synthetic_reader_config
        reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
            reader, omitter,
            "Pomegranate - Synthetic" + str(j + 1))
        _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                   "Ground Truth" + str(j + 1))
        reader_config.transmat_mode = transmat_mode

        for i, p in enumerate(probabilities):
            omitter_bernoulli_config.prob_of_observation = p  # change the prob
            x = []
            for k, power in enumerate(powers):
                reader_config.set_temporal = True
                reader_config.matrix_power = power

                pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                           pipeline_pome_config)
                print(f"finished creating {pipeline_pome_syn}")
                pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                         pipeline_gt_config)
                print(f"finished creating {pipeline_gt_syn}")

                print("matrices compared l1 norm")
                results[j, i, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_pome_syn.transmat)
                x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

            axes1[i].plot(x, results[j, i, :], marker='.', label=f"run #{j}", linestyle=":")
            axes1[i].set_title(f"p={p}")

    handles, labels = axes1[0].get_legend_handles_labels()
    fig1.legend(handles, labels, loc='lower left')
    fig1.suptitle("Pome " + omitter + " " + transmat_mode)
    fig1.subplots_adjust(hspace=0.6)
    fig1.supxlabel("Temporal Info Ratio")
    if show:
        plt.show()


def compare_pipelines_for_different_prob_transmat_mode_vs_temporal_info(omitter, n_run, powers, transmat_mode=None,
                                                                        n_probabilities=None, probabilities=None,
                                                                        show=True):
    if n_probabilities is None:
        n_probabilities = len(probabilities)
    if probabilities is None:
        probabilities = np.linspace(0.1, 1, n_probabilities)
    results = np.empty((2, n_run, n_probabilities, len(powers)))

    fig1, axes1 = plt.subplots(n_probabilities, 1, sharex=True, sharey=True, figsize=(10, 6))

    fig2, axes2 = plt.subplots(n_probabilities, 1, sharex=True, sharey=True, figsize=(10, 6))
    reader = "My Synthetic"
    for j in range(n_run):
        omitter_bernoulli_config: bernoulli_omitter_config
        reader_config: my_synthetic_reader_config
        reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
            reader, omitter,
            "Pomegranate - Synthetic" + str(j + 1))
        _, _, pipeline_gibbs_config = create_config(reader, "Bernoulli",
                                                    "Gibbs Sampler" + str(j + 1))
        _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                   "Ground Truth" + str(j + 1))
        reader_config.transmat_mode = transmat_mode

        for i, p in enumerate(probabilities):
            omitter_bernoulli_config.prob_of_observation = p  # change the prob
            x = []
            for k, power in enumerate(powers):
                reader_config.set_temporal = True
                reader_config.matrix_power = power

                pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                           pipeline_pome_config)
                print(f"finished creating {pipeline_pome_syn}")
                pipeline_gibbs_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                            pipeline_gibbs_config)
                print(f"finished creating {pipeline_gibbs_syn}")
                pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                         pipeline_gt_config)
                print(f"finished creating {pipeline_gt_syn}")

                print("matrices compared l1 norm")
                results[0, j, i, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_pome_syn.transmat)
                results[1, j, i, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_gibbs_syn.transmat)
                x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

            axes1[i].plot(x, results[0, j, i, :], marker='.', label=f"run #{j}", linestyle=":")
            axes1[i].set_title(f"p={p}")

            axes2[i].plot(x, results[1, j, i, :], marker='.', label=f"run #{j}", linestyle=":")
            axes2[i].set_title(f"p={p}")

    handles, labels = axes1[0].get_legend_handles_labels()
    fig1.legend(handles, labels, loc='lower left')
    fig1.suptitle("Pome " + omitter + " " + transmat_mode)
    fig1.subplots_adjust(hspace=0.6)
    fig1.supxlabel("Temporal Info Ratio")
    handles, labels = axes2[0].get_legend_handles_labels()
    fig2.legend(handles, labels, loc='lower left')
    fig2.suptitle("Gibbs " + omitter + " " + transmat_mode)
    fig2.subplots_adjust(hspace=0.6)
    fig2.supxlabel("Temporal Info Ratio")
    if show:
        plt.show()


def run_simple_test(reader_name, omitter_name, pipeline_name):
    fitted_pipeline: pipeline = create_pipeline(reader_name, omitter_name, pipeline_name)
    print(f"finished creating {fitted_pipeline}")
    print(fitted_pipeline.transmat)


def run_simple_test_transmat_mode(reader_name, omitter_name, pipeline_name, transmat_mode):
    reader_config, omitter_config, pipeline_config = create_config(reader_name, omitter_name, pipeline_name)
    reader_config.transmat_mode = transmat_mode
    fitted_pipeline: pipeline = create_pipeline_from_configs(reader_config, omitter_config, pipeline_config)
    print(f"finished creating {fitted_pipeline}")
    print(fitted_pipeline.transmat)


def compare_pipelines_for_different_prob_transmat_mode_with_temp_info_vs_prob(omitter, n_run, powers,
                                                                              transmat_mode=None,
                                                                              n_probabilities=None, probabilities=None,
                                                                              show=True):
    if n_probabilities is None:
        n_probabilities = len(probabilities)
    if probabilities is None:
        probabilities = np.linspace(0.1, 1, n_probabilities)
    results = np.empty((2, n_run, n_probabilities, len(powers)))
    reader = "My Synthetic"
    for j in range(n_run):
        omitter_bernoulli_config: bernoulli_omitter_config
        reader_config: my_synthetic_reader_config
        reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
            reader, omitter,
            "Pomegranate - Synthetic" + str(j + 1))
        _, _, pipeline_gibbs_config = create_config(reader, omitter,
                                                    "Gibbs Sampler" + str(j + 1))
        _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                   "Ground Truth" + str(j + 1))
        reader_config.transmat_mode = transmat_mode

        for i, p in enumerate(probabilities):
            omitter_bernoulli_config.prob_of_observation = p  # change the prob
            x = []
            for k, power in enumerate(powers):
                reader_config.set_temporal = True
                reader_config.matrix_power = power

                pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                           pipeline_pome_config)
                print(f"finished creating {pipeline_pome_syn}")
                pipeline_gibbs_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                            pipeline_gibbs_config)
                print(f"finished creating {pipeline_gibbs_syn}")
                pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                         pipeline_gt_config)
                print(f"finished creating {pipeline_gt_syn}")

                print("matrices compared l1 norm")
                results[0, j, i, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat,
                                                          pipeline_pome_syn.transmat)
                results[1, j, i, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat,
                                                          pipeline_gibbs_syn.transmat)
                x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

    results = np.average(results, axis=1)
    cmap = cm.get_cmap('plasma')
    for k, power in enumerate(powers):
        color = cmap(x[k])
        plt.figure(1)
        plt.plot(probabilities, results[0, :, k], marker='.', label=f"temp info = {x[k]:.2f}", linestyle=":",
                 color=color)

        plt.figure(2)
        plt.plot(probabilities, results[1, :, k], marker='.', label=f"temp info = {x[k]:.2f}", linestyle=":",
                 color=color)

    plt.figure(1)
    plt.legend()
    plt.title("Pome " + omitter + " " + transmat_mode)
    plt.xlabel("probability")
    plt.ylabel("L1")
    plt.figure(2)
    plt.legend()
    plt.title("Gibbs " + omitter + " " + transmat_mode)
    plt.xlabel("probability")
    plt.ylabel("L1")
    if show:
        plt.show()


def compare_pipelines_for_different_prob_transmat_mode_vs_temporal_info(n_run, powers, transmat_modes, show=True):
    omitter = "Bernoulli"
    reader = "My Synthetic"
    for i, transmat_mode in enumerate(transmat_modes):
        results = np.empty((2, n_run, len(powers[i])))
        for j in range(n_run):
            omitter_bernoulli_config: bernoulli_omitter_config
            reader_config: my_synthetic_reader_config
            reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
                reader, omitter,
                "Pomegranate - Synthetic" + str(j + 1))
            _, _, pipeline_gibbs_config = create_config(reader, "Bernoulli",
                                                        "Gibbs Sampler" + str(j + 1))
            _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                       "Ground Truth" + str(j + 1))
            reader_config.transmat_mode = transmat_mode
            omitter_bernoulli_config.prob_of_observation = 1.0  # pass all
            x = []
            for k, power in enumerate(powers[i]):
                reader_config.set_temporal = True
                reader_config.matrix_power = power

                pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                           pipeline_pome_config)
                print(f"finished creating {pipeline_pome_syn}")
                pipeline_gibbs_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                            pipeline_gibbs_config)
                print(f"finished creating {pipeline_gibbs_syn}")
                pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                         pipeline_gt_config)
                print(f"finished creating {pipeline_gt_syn}")

                print("matrices compared l1 norm")

                results[0, j, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_pome_syn.transmat)
                results[1, j, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_gibbs_syn.transmat)
                x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

        results = np.average(results, axis=1)
        plt.figure(1, figsize=(height, width))
        plt.plot(x, results[0], marker='.', label=f"Transition Mode = {transmat_mode}", linestyle=":")

        plt.figure(2, figsize=(height, width))
        plt.plot(x, results[1], marker='.', label=f"Transition Mode = {transmat_mode}", linestyle=":")

    plt.figure(1)
    plt.legend()
    plt.title("Pome Pass All")
    plt.xlabel("Temporal Info")
    plt.ylabel("L1")
    plt.savefig("Pome Pass All")
    plt.figure(2)
    plt.legend()
    plt.title("Gibbs Pass All")
    plt.xlabel("Temporal Info")
    plt.ylabel("L1")
    plt.savefig("Gibbs Pass All")
    if show:
        plt.show()


def compare_pipelines_for_different_prob_transmat_mode_vs_temporal_info_norm(n_run, powers, transmat_modes, show=True):
    omitter = "Bernoulli"
    reader = "My Synthetic"
    for i, transmat_mode in enumerate(transmat_modes):
        results = np.empty((2, n_run, len(powers[i])))
        for j in range(n_run):
            omitter_bernoulli_config: bernoulli_omitter_config
            reader_config: my_synthetic_reader_config
            reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
                reader, omitter,
                "Pomegranate - Synthetic" + str(j + 1))
            _, _, pipeline_gibbs_config = create_config(reader, "Bernoulli",
                                                        "Gibbs Sampler" + str(j + 1))
            _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                       "Ground Truth" + str(j + 1))
            reader_config.transmat_mode = transmat_mode
            omitter_bernoulli_config.prob_of_observation = 1.0  # pass all
            x = []
            for k, power in enumerate(powers[i]):
                reader_config.set_temporal = True
                reader_config.matrix_power = power

                pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                           pipeline_pome_config)
                print(f"finished creating {pipeline_pome_syn}")
                pipeline_gibbs_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                            pipeline_gibbs_config)
                print(f"finished creating {pipeline_gibbs_syn}")
                pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                         pipeline_gt_config)
                print(f"finished creating {pipeline_gt_syn}")

                print("matrices compared l1 norm")
                pome_stochastic_value = compare_mat_l1_norm(pipeline_gt_syn.transmat_list[0],
                                                            pipeline_pome_syn.transmat_list[0])
                gibbs_stochastic_value = compare_mat_l1_norm(pipeline_gt_syn.transmat_list[0],
                                                             pipeline_gibbs_syn.transmat_list[0])
                results[0, j, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_pome_syn.transmat) / \
                                   pome_stochastic_value
                results[1, j, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_gibbs_syn.transmat) / \
                                   gibbs_stochastic_value
                x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

        results = np.average(results, axis=1)
        plt.figure(1, figsize=(height, width))
        plt.plot(x, results[0], marker='.', label=f"Transition Mode = {transmat_mode}", linestyle=":")

        plt.figure(2, figsize=(height, width))
        plt.plot(x, results[1], marker='.', label=f"Transition Mode = {transmat_mode}", linestyle=":")

    plt.figure(1)
    plt.legend()
    plt.title("Pome Pass All")
    plt.xlabel("Temporal Info")
    plt.ylabel("L1")
    plt.savefig("Pome Pass All" + "Norm")
    plt.figure(2)
    plt.legend()
    plt.title("Gibbs Pass All")
    plt.xlabel("Temporal Info")
    plt.ylabel("L1")
    plt.savefig("Gibbs Pass All" + "Norm")
    if show:
        plt.show()


def compare_results_from_different_omitters(n_run, powers, transmat_mode, omitters, probabilities, show=True):
    reader = "My Synthetic"
    results = np.empty((2, n_run, len(probabilities), len(powers)))
    for i, omitter in enumerate(omitters):
        for j in range(n_run):
            reader_config: my_synthetic_reader_config
            reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
                reader, omitter,
                "Pomegranate - Synthetic" + str(j + 1))
            _, _, pipeline_gibbs_config = create_config(reader, omitter,
                                                        "Gibbs Sampler" + str(j + 1))
            _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                       "Ground Truth" + str(j + 1))
            reader_config.transmat_mode = transmat_mode
            omitter_bernoulli_config.prob_of_observation = probabilities[i]
            x = []
            for k, power in enumerate(powers):
                reader_config.set_temporal = True
                reader_config.matrix_power = power

                pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                           pipeline_pome_config)
                print(f"finished creating {pipeline_pome_syn}")
                pipeline_gibbs_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                            pipeline_gibbs_config)
                print(f"finished creating {pipeline_gibbs_syn}")
                pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                         pipeline_gt_config)
                print(f"finished creating {pipeline_gt_syn}")

                print("matrices compared l1 norm")
                results[0, j, i, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_pome_syn.transmat)
                results[1, j, i, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_gibbs_syn.transmat)
                x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

        averaged_results = np.average(results, axis=1)
        plt.figure(1, figsize=(height, width))
        plt.plot(x, averaged_results[0, i], marker='.', label=f"Omitter = {omitter}", linestyle=":")

        plt.figure(2, figsize=(height, width))
        plt.plot(x, averaged_results[1, i], marker='.', label=f"Omitter = {omitter}", linestyle=":")

    plt.figure(1)
    plt.legend()
    plt.title("Pome Half Seen " + transmat_mode)
    plt.xlabel("Temporal Info")
    plt.ylabel("L1")
    plt.figure(2)
    plt.legend()
    plt.title("Gibbs Half Seen " + transmat_mode)
    plt.xlabel("Temporal Info")
    plt.ylabel("L1")
    if show:
        plt.show()


def compare_results_from_different_omitters_transmat_modes(n_run, powers, transmat_modes, omitters, probabilities,
                                                           show=True):
    reader = "My Synthetic"
    fig1, axes1 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))
    fig2, axes2 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))

    for l, transmat_mode in enumerate(transmat_modes):
        results = np.empty((2, n_run, len(probabilities), len(powers[l])))

        for i, omitter in enumerate(omitters):
            for j in range(n_run):
                reader_config: my_synthetic_reader_config
                reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
                    reader, omitter,
                    "Pomegranate - Synthetic" + str(j + 1))
                _, _, pipeline_gibbs_config = create_config(reader, omitter,
                                                            "Gibbs Sampler" + str(j + 1))
                _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                           "Ground Truth" + str(j + 1))
                reader_config.transmat_mode = transmat_mode
                omitter_bernoulli_config.prob_of_observation = probabilities[i]
                x = []
                for k, power in enumerate(powers[l]):
                    reader_config.set_temporal = True
                    reader_config.matrix_power = power

                    pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                               pipeline_pome_config)
                    print(f"finished creating {pipeline_pome_syn}")
                    pipeline_gibbs_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                                pipeline_gibbs_config)
                    print(f"finished creating {pipeline_gibbs_syn}")
                    pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                             pipeline_gt_config)
                    print(f"finished creating {pipeline_gt_syn}")

                    print("matrices compared l1 norm")
                    results[0, j, i, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_pome_syn.transmat)
                    results[1, j, i, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_gibbs_syn.transmat)
                    x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

        averaged_results = np.average(results, axis=1)
        for i, omitter in enumerate(omitters):
            axes1[l].plot(x, averaged_results[0, i], marker='.', label=f"Omitter = {omitter}", linestyle=":")
            axes1[l].set_ylabel("L1")
            axes1[l].set_title(transmat_mode)

            axes2[l].plot(x, averaged_results[1, i], marker='.', label=f"Omitter = {omitter}", linestyle=":")
            axes2[l].set_ylabel("L1")
            axes2[l].set_title(transmat_mode)
        axes1[l].legend()
        axes2[l].legend()

    fig1.suptitle("Pome 50 Percent")
    fig1.supxlabel("temporal information")
    fig1.savefig("Pome 50 Percent")
    fig2.suptitle("Gibbs 50 Percent")
    fig2.supxlabel("temporal information")
    fig2.savefig("Gibbs 50 Percent")
    if show:
        plt.show()


def compare_results_from_different_omitters_transmat_modes_norm(n_run, powers, transmat_modes, omitters, probabilities,
                                                                show=True):
    reader = "My Synthetic"
    fig1, axes1 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))
    fig2, axes2 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))

    for l, transmat_mode in enumerate(transmat_modes):
        results = np.empty((2, n_run, len(probabilities), len(powers[l])))

        for i, omitter in enumerate(omitters):
            for j in range(n_run):
                reader_config: my_synthetic_reader_config
                reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
                    reader, omitter,
                    "Pomegranate - Synthetic" + str(j + 1))
                _, _, pipeline_gibbs_config = create_config(reader, omitter,
                                                            "Gibbs Sampler" + str(j + 1))
                _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                           "Ground Truth" + str(j + 1))
                reader_config.transmat_mode = transmat_mode
                omitter_bernoulli_config.prob_of_observation = probabilities[i]
                x = []
                for k, power in enumerate(powers[l]):
                    reader_config.set_temporal = True
                    reader_config.matrix_power = power

                    pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                               pipeline_pome_config)
                    print(f"finished creating {pipeline_pome_syn}")
                    pipeline_gibbs_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                                pipeline_gibbs_config)
                    print(f"finished creating {pipeline_gibbs_syn}")
                    pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                             pipeline_gt_config)
                    print(f"finished creating {pipeline_gt_syn}")

                    print("matrices compared l1 norm")
                    pome_stochastic_value = compare_mat_l1_norm(pipeline_gt_syn.transmat_list[0],
                                                                pipeline_pome_syn.transmat_list[0])
                    gibbs_stochastic_value = compare_mat_l1_norm(pipeline_gt_syn.transmat_list[0],
                                                                 pipeline_gibbs_syn.transmat_list[0])
                    results[0, j, i, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_pome_syn.transmat) / \
                                          pome_stochastic_value
                    results[1, j, i, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_gibbs_syn.transmat) / \
                                          gibbs_stochastic_value
                    x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

        averaged_results = np.average(results, axis=1)
        for i, omitter in enumerate(omitters):
            axes1[l].plot(x, averaged_results[0, i], marker='.', label=f"Omitter = {omitter}", linestyle=":")
            axes1[l].set_ylabel("L1")
            axes1[l].set_title(transmat_mode)

            axes2[l].plot(x, averaged_results[1, i], marker='.', label=f"Omitter = {omitter}", linestyle=":")
            axes2[l].set_ylabel("L1")
            axes2[l].set_title(transmat_mode)
        axes1[l].legend()
        axes2[l].legend()

    fig1.suptitle("Pome 50 Percent")
    fig1.supxlabel("temporal information")
    fig1.savefig("Pome 50 Percent" + "Norm")
    fig2.suptitle("Gibbs 50 Percent")
    fig2.supxlabel("temporal information")
    fig2.savefig("Gibbs 50 Percent" + "Norm")
    if show:
        plt.show()


def compare_pipelines_for_different_prob_transmat_modes_with_temp_info_vs_prob(omitter, n_run, powers,
                                                                               transmat_modes=None,
                                                                               n_probabilities=None, probabilities=None,
                                                                               show=True):
    fig1, axes1 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))
    fig2, axes2 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))
    for l, transmat_mode in enumerate(transmat_modes):
        if n_probabilities is None:
            n_probabilities = len(probabilities)
        if probabilities is None:
            probabilities = np.linspace(0.1, 1, n_probabilities)
        results = np.empty((2, n_run, n_probabilities, len(powers[l])))
        reader = "My Synthetic"
        for j in range(n_run):
            omitter_bernoulli_config: bernoulli_omitter_config
            reader_config: my_synthetic_reader_config
            reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
                reader, omitter,
                "Pomegranate - Synthetic" + str(j + 1))
            _, _, pipeline_gibbs_config = create_config(reader, "Bernoulli",
                                                        "Gibbs Sampler" + str(j + 1))
            _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                       "Ground Truth" + str(j + 1))
            reader_config.transmat_mode = transmat_mode

            for i, p in enumerate(probabilities):
                omitter_bernoulli_config.prob_of_observation = p  # change the prob
                x = []
                for k, power in enumerate(powers[l]):
                    reader_config.set_temporal = True
                    reader_config.matrix_power = power

                    pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                               pipeline_pome_config)
                    print(f"finished creating {pipeline_pome_syn}")
                    pipeline_gibbs_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                                pipeline_gibbs_config)
                    print(f"finished creating {pipeline_gibbs_syn}")
                    pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                             pipeline_gt_config)
                    print(f"finished creating {pipeline_gt_syn}")

                    print("matrices compared l1 norm")
                    results[0, j, i, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat,
                                                              pipeline_pome_syn.transmat)
                    results[1, j, i, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat,
                                                              pipeline_gibbs_syn.transmat)
                    x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

        results = np.average(results, axis=1)
        cmap = cm.get_cmap('viridis')
        for k, power in enumerate(powers[l]):
            color = cmap(x[k] / max(x))  # Normalize x[k] to [0, 1]
            axes1[l].plot(probabilities, results[0, :, k], marker='.', label=f"temp info = {x[k]:.2f}", linestyle=":",
                          color=color)
            axes1[l].set_title(transmat_mode)
            axes1[l].set_ylabel("L1")

            axes2[l].plot(probabilities, results[1, :, k], marker='.', label=f"temp info = {x[k]:.2f}", linestyle=":",
                          color=color)
            axes2[l].set_title(transmat_mode)
            axes2[l].set_ylabel("L1")
        axes1[l].legend()
        axes2[l].legend()

    fig1.suptitle("Pome " + omitter)
    fig1.supxlabel("Probability")
    fig1.savefig("Pome" + omitter)

    fig2.suptitle("Gibbs " + omitter)
    fig2.supxlabel("Probability")
    fig2.savefig("Gibbs" + omitter)

    if show:
        plt.show()


def compare_pipelines_for_different_prob_transmat_modes_with_temp_info_vs_prob_norm(omitter, n_run, powers,
                                                                                    transmat_modes=None,
                                                                                    n_probabilities=None,
                                                                                    probabilities=None,
                                                                                    show=True):
    fig1, axes1 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))
    fig2, axes2 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))
    for l, transmat_mode in enumerate(transmat_modes):
        if n_probabilities is None:
            n_probabilities = len(probabilities)
        if probabilities is None:
            probabilities = np.linspace(0.1, 1, n_probabilities)
        results = np.empty((2, n_run, n_probabilities, len(powers[l])))
        reader = "My Synthetic"
        for j in range(n_run):
            omitter_bernoulli_config: bernoulli_omitter_config
            reader_config: my_synthetic_reader_config
            reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
                reader, omitter,
                "Pomegranate - Synthetic" + str(j + 1))
            _, _, pipeline_gibbs_config = create_config(reader, "Bernoulli",
                                                        "Gibbs Sampler" + str(j + 1))
            _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                       "Ground Truth" + str(j + 1))
            reader_config.transmat_mode = transmat_mode

            for i, p in enumerate(probabilities):
                omitter_bernoulli_config.prob_of_observation = p  # change the prob
                x = []
                for k, power in enumerate(powers[l]):
                    reader_config.set_temporal = True
                    reader_config.matrix_power = power

                    pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                               pipeline_pome_config)
                    print(f"finished creating {pipeline_pome_syn}")
                    pipeline_gibbs_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                                pipeline_gibbs_config)
                    print(f"finished creating {pipeline_gibbs_syn}")
                    pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                             pipeline_gt_config)
                    print(f"finished creating {pipeline_gt_syn}")

                    print("matrices compared l1 norm")
                    pome_stochastic_value = compare_mat_l1_norm(pipeline_gt_syn.transmat_list[0],
                                                                pipeline_pome_syn.transmat_list[0])
                    gibbs_stochastic_value = compare_mat_l1_norm(pipeline_gt_syn.transmat_list[0],
                                                                 pipeline_gibbs_syn.transmat_list[0])
                    results[0, j, i, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat,
                                                              pipeline_pome_syn.transmat) / pome_stochastic_value
                    results[1, j, i, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat,
                                                              pipeline_gibbs_syn.transmat) / gibbs_stochastic_value
                    x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

        results = np.average(results, axis=1)
        cmap = cm.get_cmap('viridis')
        for k, power in enumerate(powers[l]):
            color = cmap(x[k] / max(x))  # Normalize x[k] to [0, 1]
            axes1[l].plot(probabilities, results[0, :, k], marker='.', label=f"temp info = {x[k]:.2f}", linestyle=":",
                          color=color)
            axes1[l].set_title(transmat_mode)
            axes1[l].set_ylabel("L1")

            axes2[l].plot(probabilities, results[1, :, k], marker='.', label=f"temp info = {x[k]:.2f}", linestyle=":",
                          color=color)
            axes2[l].set_title(transmat_mode)
            axes2[l].set_ylabel("L1")
        axes1[l].legend()
        axes2[l].legend()

    fig1.suptitle("Pome " + omitter)
    fig1.supxlabel("Probability")
    fig1.savefig("Pome" + omitter + "Norm")

    fig2.suptitle("Gibbs " + omitter)
    fig2.supxlabel("Probability")
    fig2.savefig("Gibbs" + omitter + "Norm")

    if show:
        plt.show()


def compare_pipelines_transmat_mode_with_vs_temp_info_var(omitter, n_run, powers,
                                                          transmat_modes=None,
                                                          show=True):
    fig1, axes1 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))
    fig2, axes2 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))
    for l, transmat_mode in enumerate(transmat_modes):
        results = np.empty((2, n_run, len(powers[l])))
        reader = "My Synthetic"
        for j in range(n_run):
            reader_config: my_synthetic_reader_config
            reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
                reader, omitter,
                "Pomegranate - Synthetic" + str(j + 1))
            _, _, pipeline_gibbs_config = create_config(reader, omitter,
                                                        "Gibbs Sampler" + str(j + 1))
            _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                       "Ground Truth" + str(j + 1))
            reader_config.transmat_mode = transmat_mode

            x = []
            for k, power in enumerate(powers[l]):
                reader_config.set_temporal = True
                reader_config.matrix_power = power

                pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                           pipeline_pome_config)
                print(f"finished creating {pipeline_pome_syn}")
                pipeline_gibbs_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                            pipeline_gibbs_config)
                print(f"finished creating {pipeline_gibbs_syn}")
                pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                         pipeline_gt_config)
                print(f"finished creating {pipeline_gt_syn}")

                print("matrices compared l1 norm")
                results[0, j, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat,
                                                       pipeline_pome_syn.transmat)
                results[1, j, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat,
                                                       pipeline_gibbs_syn.transmat)
                x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

        for j in range(n_run):
            axes1[l].plot(x, results[0, j], marker='.', label=f"run #{j + 1}", linestyle=":", )
            axes1[l].set_title(transmat_mode)
            axes1[l].set_ylabel("L1")

            axes2[l].plot(x, results[1, j], marker='.', label=f"run #{j + 1}", linestyle=":", )
            axes2[l].set_title(transmat_mode)
            axes2[l].set_ylabel("L1")
        axes1[l].legend()
        axes2[l].legend()

    fig1.suptitle("Pome " + omitter)
    fig1.supxlabel("temporal information")
    fig1.savefig("Pome " + omitter + " non avg")
    plt.savefig("")
    fig2.suptitle("Gibbs " + omitter)
    fig2.supxlabel("temporal information")
    fig2.savefig("Gibbs " + omitter + " non avg")
    if show:
        plt.show()


def compare_pipelines_transmat_mode_with_vs_temp_info_var_norm(omitter, n_run, powers,
                                                               transmat_modes=None,
                                                               show=True):
    fig1, axes1 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))
    fig2, axes2 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))
    for l, transmat_mode in enumerate(transmat_modes):
        results = np.empty((2, n_run, len(powers[l])))
        reader = "My Synthetic"
        for j in range(n_run):
            reader_config: my_synthetic_reader_config
            reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
                reader, omitter,
                "Pomegranate - Synthetic" + str(j + 1))
            _, _, pipeline_gibbs_config = create_config(reader, omitter,
                                                        "Gibbs Sampler" + str(j + 1))
            _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                       "Ground Truth" + str(j + 1))
            reader_config.transmat_mode = transmat_mode

            x = []
            for k, power in enumerate(powers[l]):
                reader_config.set_temporal = True
                reader_config.matrix_power = power

                pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                           pipeline_pome_config)
                print(f"finished creating {pipeline_pome_syn}")
                pipeline_gibbs_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                            pipeline_gibbs_config)
                print(f"finished creating {pipeline_gibbs_syn}")
                pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                         pipeline_gt_config)
                print(f"finished creating {pipeline_gt_syn}")

                print("matrices compared l1 norm")
                pome_stochastic_value = compare_mat_l1_norm(pipeline_gt_syn.transmat_list[0],
                                                            pipeline_pome_syn.transmat_list[0])
                gibbs_stochastic_value = compare_mat_l1_norm(pipeline_gt_syn.transmat_list[0],
                                                             pipeline_gibbs_syn.transmat_list[0])
                results[0, j, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat,
                                                       pipeline_pome_syn.transmat) / pome_stochastic_value
                results[1, j, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat,
                                                       pipeline_gibbs_syn.transmat) / gibbs_stochastic_value
                x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

        for j in range(n_run):
            axes1[l].plot(x, results[0, j], marker='.', label=f"run #{j + 1}", linestyle=":", )
            axes1[l].set_title(transmat_mode)
            axes1[l].set_ylabel("L1")

            axes2[l].plot(x, results[1, j], marker='.', label=f"run #{j + 1}", linestyle=":", )
            axes2[l].set_title(transmat_mode)
            axes2[l].set_ylabel("L1")
        axes1[l].legend()
        axes2[l].legend()

    fig1.suptitle("Pome " + omitter)
    fig1.supxlabel("temporal information")
    fig1.savefig("Pome " + omitter + " non avg" + "Norm")
    plt.savefig("")
    fig2.suptitle("Gibbs " + omitter)
    fig2.supxlabel("temporal information")
    fig2.savefig("Gibbs " + omitter + " non avg" + "Norm")
    if show:
        plt.show()


def compare_pipelines_transmat_mode_with_vs_temp_info_avg(omitter, n_run, powers,
                                                          transmat_modes=None,
                                                          show=True):
    fig1, axes1 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))
    fig2, axes2 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))
    for l, transmat_mode in enumerate(transmat_modes):
        results = np.empty((2, n_run, len(powers[l])))
        reader = "My Synthetic"
        for j in range(n_run):
            reader_config: my_synthetic_reader_config
            reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
                reader, omitter,
                "Pomegranate - Synthetic" + str(j + 1))
            _, _, pipeline_gibbs_config = create_config(reader, omitter,
                                                        "Gibbs Sampler" + str(j + 1))
            _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                       "Ground Truth" + str(j + 1))
            reader_config.transmat_mode = transmat_mode

            x = []
            for k, power in enumerate(powers[l]):
                reader_config.set_temporal = True
                reader_config.matrix_power = power

                pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                           pipeline_pome_config)
                print(f"finished creating {pipeline_pome_syn}")
                pipeline_gibbs_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                            pipeline_gibbs_config)
                print(f"finished creating {pipeline_gibbs_syn}")
                pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                         pipeline_gt_config)
                print(f"finished creating {pipeline_gt_syn}")

                print("matrices compared l1 norm")
                results[0, j, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat,
                                                       pipeline_pome_syn.transmat)
                results[1, j, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat,
                                                       pipeline_gibbs_syn.transmat)
                x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

        results = np.average(results, axis=1)
        axes1[l].plot(x, results[0], marker='.', label=f"run #{j + 1}", linestyle=":", )
        axes1[l].set_title(transmat_mode)
        axes1[l].set_ylabel("L1")

        axes2[l].plot(x, results[1], marker='.', label=f"run #{j + 1}", linestyle=":", )
        axes2[l].set_title(transmat_mode)
        axes2[l].set_ylabel("L1")
        axes1[l].legend()
        axes2[l].legend()

    fig1.suptitle("Pome " + omitter)
    fig1.supxlabel("temporal information")
    fig1.savefig("Pome " + omitter + " avg")
    plt.savefig("")
    fig2.suptitle("Gibbs " + omitter)
    fig2.supxlabel("temporal information")
    fig2.savefig("Gibbs " + omitter + " avg")
    if show:
        plt.show()


def compare_pipelines_transmat_mode_with_vs_temp_info_avg_norm(omitter, n_run, powers,
                                                               transmat_modes=None,
                                                               show=True):
    fig1, axes1 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))
    fig2, axes2 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))
    for l, transmat_mode in enumerate(transmat_modes):
        results = np.empty((2, n_run, len(powers[l])))
        reader = "My Synthetic"
        for j in range(n_run):
            reader_config: my_synthetic_reader_config
            omitter_bernoulli_config: uniform_skips_omitter_config
            reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
                reader, omitter,
                "Pomegranate - Synthetic" + str(j + 1))
            _, _, pipeline_gibbs_config = create_config(reader, omitter,
                                                        "Gibbs Sampler" + str(j + 1))
            _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                       "Ground Truth" + str(j + 1))
            reader_config.transmat_mode = transmat_mode

            reader_config.sentence_length = int(
                reader_config.sentence_length / get_data_percent(omitter, omitter_bernoulli_config.n_skips))
            x = []
            for k, power in enumerate(powers[l]):
                reader_config.set_temporal = True
                reader_config.matrix_power = power

                pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                           pipeline_pome_config)
                print(f"finished creating {pipeline_pome_syn}")
                pipeline_gibbs_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                            pipeline_gibbs_config)
                print(f"finished creating {pipeline_gibbs_syn}")
                pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                         pipeline_gt_config)
                print(f"finished creating {pipeline_gt_syn}")

                print("matrices compared l1 norm")
                pome_stochastic_value = compare_mat_l1_norm(pipeline_gt_syn.transmat_list[0],
                                                            pipeline_pome_syn.transmat_list[0])
                gibbs_stochastic_value = compare_mat_l1_norm(pipeline_gt_syn.transmat_list[0],
                                                             pipeline_gibbs_syn.transmat_list[0])
                results[0, j, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat,
                                                       pipeline_pome_syn.transmat) / pome_stochastic_value
                results[1, j, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat,
                                                       pipeline_gibbs_syn.transmat) / gibbs_stochastic_value
                x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

        results = np.average(results, axis=1)
        axes1[l].plot(x, results[0], marker='.', linestyle=":", )
        axes1[l].set_title(transmat_mode)
        axes1[l].set_ylabel("L1")

        axes2[l].plot(x, results[1], marker='.', linestyle=":", )
        axes2[l].set_title(transmat_mode)
        axes2[l].set_ylabel("L1")

    fig1.suptitle("Pome " + omitter)
    fig1.supxlabel("temporal information")
    fig1.savefig("Pome " + omitter + " avg" + "Norm")
    fig2.suptitle("Gibbs " + omitter)
    fig2.supxlabel("temporal information")
    fig2.savefig("Gibbs " + omitter + " avg" + "Norm")
    if show:
        plt.show()


def compare_pipelines_for_different_prob_transmat_modes_with_temp_info_vs_prob_norm_pome(omitter, n_run, powers,
                                                                                         transmat_modes=None,
                                                                                         n_probabilities=None,
                                                                                         probabilities=None,
                                                                                         show=True):
    fig1, axes1 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))
    for l, transmat_mode in enumerate(transmat_modes):
        if n_probabilities is None:
            n_probabilities = len(probabilities)
        if probabilities is None:
            probabilities = np.linspace(0.1, 1, n_probabilities)
        results = np.empty((2, n_run, n_probabilities, len(powers[l])))
        reader = "My Synthetic"
        for j in range(n_run):
            omitter_bernoulli_config: bernoulli_omitter_config
            reader_config: my_synthetic_reader_config
            reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
                reader, omitter,
                "Pomegranate - Synthetic" + str(j + 1))
            _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                       "Ground Truth" + str(j + 1))
            reader_config.transmat_mode = transmat_mode

            for i, p in enumerate(probabilities):
                omitter_bernoulli_config.prob_of_observation = p  # change the prob
                x = []
                for k, power in enumerate(powers[l]):
                    reader_config.set_temporal = True
                    reader_config.matrix_power = power

                    pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                               pipeline_pome_config)
                    print(f"finished creating {pipeline_pome_syn}")
                    pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                             pipeline_gt_config)
                    print(f"finished creating {pipeline_gt_syn}")

                    print("matrices compared l1 norm")
                    pome_stochastic_value = compare_mat_l1_norm(pipeline_gt_syn.transmat_list[0],
                                                                pipeline_pome_syn.transmat_list[0])
                    results[0, j, i, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat,
                                                              pipeline_pome_syn.transmat) / pome_stochastic_value
                    x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

        results = np.average(results, axis=1)
        cmap = cm.get_cmap('viridis')
        for k, power in enumerate(powers[l]):
            color = cmap(x[k] / max(x))  # Normalize x[k] to [0, 1]
            axes1[l].plot(probabilities, results[0, :, k], marker='.', label=f"temp info = {x[k]:.2f}", linestyle=":",
                          color=color)
            axes1[l].set_title(transmat_mode)
            axes1[l].set_ylabel("L1")
        axes1[l].legend()

    fig1.suptitle("Pome " + omitter)
    fig1.supxlabel("Probability")
    fig1.savefig("Pome" + omitter + "Norm")

    if show:
        plt.show()


def compare_pipelines_for_different_prob_transmat_modes_with_temp_info_vs_iter_pome(omitter, n_run, powers, n_iter,
                                                                                    transmat_mode=None,
                                                                                    n_probabilities=None,
                                                                                    probabilities=None,
                                                                                    show=True):
    if n_probabilities is None:
        n_probabilities = len(probabilities)
    if probabilities is None:
        probabilities = np.linspace(0.1, 1, n_probabilities)
    fig1, axes1 = plt.subplots(n_probabilities, 1, sharex=True, sharey=True, figsize=(height, width))

    results = np.empty((2, n_run, n_probabilities, len(powers), n_iter + 1))
    reader = "My Synthetic"
    for j in range(n_run):
        omitter_bernoulli_config: bernoulli_omitter_config
        reader_config: my_synthetic_reader_config
        reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
            reader, omitter,
            "Pomegranate - Synthetic" + str(j + 1))
        _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                   "Ground Truth" + str(j + 1))
        reader_config.transmat_mode = transmat_mode

        for i, p in enumerate(probabilities):
            omitter_bernoulli_config.prob_of_observation = p  # change the prob
            x = []
            for k, power in enumerate(powers):
                reader_config.set_temporal = True
                reader_config.matrix_power = power

                pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                           pipeline_pome_config)
                print(f"finished creating {pipeline_pome_syn}")
                pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                         pipeline_gt_config)
                print(f"finished creating {pipeline_gt_syn}")

                print("matrices compared l1 norm")
                results[0, j, i, k] = compare_mat_l1_norm_for_list(pipeline_gt_syn.transmat_list,
                                                                   pipeline_pome_syn.transmat_list)
                results[0, j, i, k] = results[0, j, i, k] / results[0, j, i, k, 0]
                x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

    results = np.average(results, axis=1)
    cmap = cm.get_cmap('viridis')

    for i, probability in enumerate(probabilities):
        for k, power in enumerate(powers):
            color = cmap(x[k] / max(x))  # Normalize x[k] to [0, 1]
            axes1[i].plot(np.arange(n_iter + 1), results[0, i, k], marker='.', label=f"temp info = {x[k]:.2f}",
                          linestyle=":",
                          color=color)
            axes1[i].set_title(f"p = {probability:.2f}")
            axes1[i].set_ylabel("L1")
        axes1[i].legend()

    fig1.suptitle("Pome " + omitter + " " + transmat_mode)
    fig1.supxlabel("# Iter")
    fig1.savefig("Pome " + omitter + " " + transmat_mode + " Iter")

    if show:
        plt.show()


def compare_pipelines_for_different_prob_transmat_modes_with_temp_info_vs_iter(omitter, n_run, powers, n_iter,
                                                                               transmat_modes=None,
                                                                               probability=None,
                                                                               show=True):
    fig1, axes1 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))
    fig2, axes2 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))
    for l, transmat_mode in enumerate(transmat_modes):
        results = np.empty((2, n_run, n_iter, len(powers[l])))
        reader = "My Synthetic"
        for j in range(n_run):
            omitter_bernoulli_config: bernoulli_omitter_config
            reader_config: my_synthetic_reader_config
            reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
                reader, omitter,
                "Pomegranate - Synthetic" + str(j + 1))
            _, _, pipeline_gibbs_config = create_config(reader, "Bernoulli",
                                                        "Gibbs Sampler" + str(j + 1))
            _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                       "Ground Truth" + str(j + 1))
            reader_config.transmat_mode = transmat_mode

            omitter_bernoulli_config.prob_of_observation = probability  # change the prob
            x = []
            for k, power in enumerate(powers[l]):
                reader_config.set_temporal = True
                reader_config.matrix_power = power

                pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                           pipeline_pome_config)
                print(f"finished creating {pipeline_pome_syn}")
                pipeline_gibbs_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                            pipeline_gibbs_config)
                print(f"finished creating {pipeline_gibbs_syn}")
                pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                         pipeline_gt_config)
                print(f"finished creating {pipeline_gt_syn}")

                print("matrices compared l1 norm")
                pome_stochastic_value = compare_mat_l1_norm(pipeline_gt_syn.transmat_list[0],
                                                            pipeline_pome_syn.transmat_list[0])
                gibbs_stochastic_value = compare_mat_l1_norm(pipeline_gt_syn.transmat_list[0],
                                                             pipeline_gibbs_syn.transmat_list[0])
                results[0, j, :, k] = np.abs(np.diff(compare_mat_l1_norm_for_list(pipeline_gt_syn.transmat_list,
                                                                                  pipeline_pome_syn.transmat_list) / pome_stochastic_value))
                results[1, j, :, k] = np.abs(np.diff(compare_mat_l1_norm_for_list(pipeline_gt_syn.transmat_list,
                                                                                  pipeline_gibbs_syn.transmat_list) / gibbs_stochastic_value))
                x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

        results = np.average(results, axis=1)
        cmap = cm.get_cmap('viridis')
        iters = np.arange(n_iter)
        for k, power in enumerate(powers[l]):
            color = cmap(x[k] / max(x))  # Normalize x[k] to [0, 1]
            axes1[l].plot(iters, results[0, :, k], marker='.', label=f"temp info = {x[k]:.2f}", linestyle=":",
                          color=color)
            axes1[l].set_title(transmat_mode)
            axes1[l].set_ylabel("L1")

            axes2[l].plot(iters, results[1, :, k], marker='.', label=f"temp info = {x[k]:.2f}", linestyle=":",
                          color=color)
            axes2[l].set_title(transmat_mode)
            axes2[l].set_ylabel("L1")
        axes1[l].legend()
        axes2[l].legend()
        axes1[l].axhline(y=0.02, color='r', linestyle='--')
        axes2[l].axhline(y=0.02, color='r', linestyle='--')

    fig1.suptitle("Pome " + omitter + f" with Prob = {probability}")
    fig1.supxlabel("# Iter")
    fig1.savefig("Pome " + omitter + f" with Prob = {probability}", format='png')

    fig2.suptitle("Gibbs " + omitter + f" with Prob = {probability}")
    fig2.supxlabel("# Iter")
    fig2.savefig("Gibbs " + omitter + f" with Prob = {probability}", format='png')

    if show:
        plt.show()


def get_data_percent(omitter, p=None):
    if omitter == "Bernoulli":
        return p
    elif omitter == "Geometric":
        return p / (1 + p)
    elif omitter == "Markov Chain":
        return p + 0.5
    elif omitter == "Consecutive Bernoulli":
        return 1 / (p + 2 * (1 - p))
    elif omitter == "Uniform Skips":
        mean_skip = 1 + np.average(np.arange(1, p + 1))
        return 1 / mean_skip


def compare_pipelines_for_different_prob_transmat_modes_with_temp_info_vs_prob_norm_data_size_norm(omitter, n_run,
                                                                                                   powers,
                                                                                                   transmat_modes=None,
                                                                                                   n_probabilities=None,
                                                                                                   probabilities=None,
                                                                                                   show=True):
    fig1, axes1 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))
    fig2, axes2 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))
    for l, transmat_mode in enumerate(transmat_modes):
        if n_probabilities is None:
            n_probabilities = len(probabilities)
        if probabilities is None:
            probabilities = np.linspace(0.1, 1, n_probabilities)
        results = np.empty((2, n_run, n_probabilities, len(powers[l])))
        reader = "My Synthetic"
        for j in range(n_run):
            omitter_bernoulli_config: bernoulli_omitter_config
            reader_config: my_synthetic_reader_config
            reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
                reader, omitter,
                "Pomegranate - Synthetic" + str(j + 1))
            _, _, pipeline_gibbs_config = create_config(reader, "Bernoulli",
                                                        "Gibbs Sampler" + str(j + 1))
            _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                       "Ground Truth" + str(j + 1))
            reader_config.transmat_mode = transmat_mode
            sentence_length = reader_config.sentence_length
            for i, p in enumerate(probabilities):
                omitter_bernoulli_config.prob_of_observation = p  # change the prob
                data_percent = get_data_percent(omitter, p)
                reader_config.sentence_length = int(sentence_length / data_percent)
                x = []
                for k, power in enumerate(powers[l]):
                    reader_config.set_temporal = True
                    reader_config.matrix_power = power

                    pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                               pipeline_pome_config)
                    print(f"finished creating {pipeline_pome_syn}")
                    pipeline_gibbs_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                                pipeline_gibbs_config)
                    print(f"finished creating {pipeline_gibbs_syn}")
                    pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                             pipeline_gt_config)
                    print(f"finished creating {pipeline_gt_syn}")

                    print("matrices compared l1 norm")
                    pome_stochastic_value = compare_mat_l1_norm(pipeline_gt_syn.transmat_list[0],
                                                                pipeline_pome_syn.transmat_list[0])
                    gibbs_stochastic_value = compare_mat_l1_norm(pipeline_gt_syn.transmat_list[0],
                                                                 pipeline_gibbs_syn.transmat_list[0])
                    results[0, j, i, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat,
                                                              pipeline_pome_syn.transmat) / pome_stochastic_value
                    results[1, j, i, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat,
                                                              pipeline_gibbs_syn.transmat) / gibbs_stochastic_value
                    x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

        results = np.average(results, axis=1)
        cmap = cm.get_cmap('viridis')
        for k, power in enumerate(powers[l]):
            color = cmap(x[k] / max(x))  # Normalize x[k] to [0, 1]
            axes1[l].plot(probabilities, results[0, :, k], marker='.', label=f"temp info = {x[k]:.2f}", linestyle=":",
                          color=color)
            axes1[l].set_title(transmat_mode)
            axes1[l].set_ylabel("L1")

            axes2[l].plot(probabilities, results[1, :, k], marker='.', label=f"temp info = {x[k]:.2f}", linestyle=":",
                          color=color)
            axes2[l].set_title(transmat_mode)
            axes2[l].set_ylabel("L1")
        axes1[l].legend()
        axes2[l].legend()

    fig1.suptitle("Pome " + omitter)
    fig1.supxlabel("Probability")
    fig1.savefig("Pome" + omitter + "Norm")

    fig2.suptitle("Gibbs " + omitter)
    fig2.supxlabel("Probability")
    fig2.savefig("Gibbs" + omitter + "Norm")

    if show:
        plt.show()


def compare_pipelines_transmat_mode_with_vs_temp_info_avg_norm_n_readers(omitter, n_run, powers, n_readers,
                                                               transmat_modes=None,
                                                               show=True):
    for i in range(n_readers):
        fig1, axes1 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))
        fig2, axes2 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))
        for l, transmat_mode in enumerate(transmat_modes):
            results = np.empty((2, n_run, len(powers[l])))
            reader = "My Synthetic" + str(i+1)
            for j in range(n_run):
                reader_config: my_synthetic_reader_config
                omitter_bernoulli_config: uniform_skips_omitter_config
                reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
                    reader, omitter,
                    "Pomegranate - Synthetic" + str(j + 1))
                _, _, pipeline_gibbs_config = create_config(reader, omitter,
                                                            "Gibbs Sampler" + str(j + 1))
                _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                           "Ground Truth" + str(j + 1))
                reader_config.transmat_mode = transmat_mode

                reader_config.sentence_length = int(
                    reader_config.sentence_length / get_data_percent(omitter, omitter_bernoulli_config.n_skips))
                x = []
                for k, power in enumerate(powers[l]):
                    reader_config.set_temporal = True
                    reader_config.matrix_power = power

                    pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                               pipeline_pome_config)
                    print(f"finished creating {pipeline_pome_syn}")
                    pipeline_gibbs_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                                pipeline_gibbs_config)
                    print(f"finished creating {pipeline_gibbs_syn}")
                    pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                             pipeline_gt_config)
                    print(f"finished creating {pipeline_gt_syn}")

                    print("matrices compared l1 norm")
                    pome_stochastic_value = compare_mat_l1_norm(pipeline_gt_syn.transmat_list[0],
                                                                pipeline_pome_syn.transmat_list[0])
                    gibbs_stochastic_value = compare_mat_l1_norm(pipeline_gt_syn.transmat_list[0],
                                                                 pipeline_gibbs_syn.transmat_list[0])
                    results[0, j, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat,
                                                           pipeline_pome_syn.transmat) / pome_stochastic_value
                    results[1, j, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat,
                                                           pipeline_gibbs_syn.transmat) / gibbs_stochastic_value
                    x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

            results = np.average(results, axis=1)
            axes1[l].plot(x, results[0], marker='.', linestyle=":", )
            axes1[l].set_title(transmat_mode)
            axes1[l].set_ylabel("L1")

            axes2[l].plot(x, results[1], marker='.', linestyle=":", )
            axes2[l].set_title(transmat_mode)
            axes2[l].set_ylabel("L1")

        fig1.suptitle("Pome " + omitter)
        fig1.supxlabel("temporal information")
        fig1.savefig("Pome " + omitter + " avg" + "Norm")
        fig2.suptitle("Gibbs " + omitter)
        fig2.supxlabel("temporal information")
        fig2.savefig("Gibbs " + omitter + " avg" + "Norm")
        if show:
            plt.show()

def compare_results_from_different_omitters_transmat_modes_norm_data_size_norm(n_run, powers, transmat_modes, omitters, probabilities,
                                                                show=True):
    reader = "My Synthetic"
    fig1, axes1 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))
    fig2, axes2 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))

    for l, transmat_mode in enumerate(transmat_modes):
        results = np.empty((2, n_run, len(probabilities), len(powers[l])))

        for i, omitter in enumerate(omitters):
            for j in range(n_run):
                reader_config: my_synthetic_reader_config
                reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
                    reader, omitter,
                    "Pomegranate - Synthetic" + str(j + 1))
                _, _, pipeline_gibbs_config = create_config(reader, omitter,
                                                            "Gibbs Sampler" + str(j + 1))
                _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                           "Ground Truth" + str(j + 1))
                reader_config.transmat_mode = transmat_mode
                omitter_bernoulli_config.prob_of_observation = probabilities[i]
                reader_config.sentence_length = int(
                    reader_config.sentence_length / get_data_percent(omitter, probabilities[i]))
                x = []
                for k, power in enumerate(powers[l]):
                    reader_config.set_temporal = True
                    reader_config.matrix_power = power

                    pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                               pipeline_pome_config)
                    print(f"finished creating {pipeline_pome_syn}")
                    pipeline_gibbs_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                                pipeline_gibbs_config)
                    print(f"finished creating {pipeline_gibbs_syn}")
                    pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                             pipeline_gt_config)
                    print(f"finished creating {pipeline_gt_syn}")

                    print("matrices compared l1 norm")
                    pome_stochastic_value = compare_mat_l1_norm(pipeline_gt_syn.transmat_list[0],
                                                                pipeline_pome_syn.transmat_list[0])
                    gibbs_stochastic_value = compare_mat_l1_norm(pipeline_gt_syn.transmat_list[0],
                                                                 pipeline_gibbs_syn.transmat_list[0])
                    results[0, j, i, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_pome_syn.transmat) / \
                                          pome_stochastic_value
                    results[1, j, i, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_gibbs_syn.transmat) / \
                                          gibbs_stochastic_value
                    x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

        averaged_results = np.average(results, axis=1)
        for i, omitter in enumerate(omitters):
            axes1[l].plot(x, averaged_results[0, i], marker='.', label=f"Omitter = {omitter}", linestyle=":")
            axes1[l].set_ylabel("L1")
            axes1[l].set_title(transmat_mode)

            axes2[l].plot(x, averaged_results[1, i], marker='.', label=f"Omitter = {omitter}", linestyle=":")
            axes2[l].set_ylabel("L1")
            axes2[l].set_title(transmat_mode)
        axes1[l].legend()
        axes2[l].legend()

    fig1.suptitle("Pome 50 Percent")
    fig1.supxlabel("temporal information")
    fig1.savefig("Pome 50 Percent" + "Norm")
    fig2.suptitle("Gibbs 50 Percent")
    fig2.supxlabel("temporal information")
    fig2.savefig("Gibbs 50 Percent" + "Norm")
    if show:
        plt.show()
def compare_pipelines_for_different_prob_transmat_modes_with_temp_info_vs_prob_norm_data_size_norm_for_pome(omitter, n_run,
                                                                                                   powers,
                                                                                                   transmat_modes=None,
                                                                                                   n_probabilities=None,
                                                                                                   probabilities=None,
                                                                                                   show=True):
    fig1, axes1 = plt.subplots(len(transmat_modes), 1, sharex=True, sharey=True, figsize=(height, width))

    for l, transmat_mode in enumerate(transmat_modes):
        if n_probabilities is None:
            n_probabilities = len(probabilities)
        if probabilities is None:
            probabilities = np.linspace(0.1, 1, n_probabilities)
        results = np.empty((2, n_run, n_probabilities, len(powers[l])))
        reader = "My Synthetic"
        for j in range(n_run):
            omitter_bernoulli_config: bernoulli_omitter_config
            reader_config: my_synthetic_reader_config
            reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
                reader, omitter,
                "Pomegranate - Synthetic" + str(j + 1))

            _, omitter_pass_config, pipeline_gt_config = create_config(reader, "Pass All",
                                                                       "Ground Truth" + str(j + 1))
            reader_config.transmat_mode = transmat_mode
            sentence_length = reader_config.sentence_length
            for i, p in enumerate(probabilities):
                omitter_bernoulli_config.prob_of_observation = p  # change the prob
                data_percent = get_data_percent(omitter, p)
                reader_config.sentence_length = int(sentence_length / data_percent)
                x = []
                for k, power in enumerate(powers[l]):
                    reader_config.set_temporal = True
                    reader_config.matrix_power = power

                    pipeline_pome_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_bernoulli_config,
                                                                               pipeline_pome_config)
                    print(f"finished creating {pipeline_pome_syn}")

                    pipeline_gt_syn: pipeline = create_pipeline_from_configs(reader_config, omitter_pass_config,
                                                                             pipeline_gt_config)
                    print(f"finished creating {pipeline_gt_syn}")

                    print("matrices compared l1 norm")
                    pome_stochastic_value = compare_mat_l1_norm(pipeline_gt_syn.transmat_list[0],
                                                                pipeline_pome_syn.transmat_list[0])

                    results[0, j, i, k] = compare_mat_l1_norm(pipeline_gt_syn.transmat,
                                                              pipeline_pome_syn.transmat) / pome_stochastic_value

                    x.append(find_temporal_info_ratio(pipeline_gt_syn.transmat))

        results = np.average(results, axis=1)
        cmap = cm.get_cmap('viridis')
        for k, power in enumerate(powers[l]):
            color = cmap(x[k] / max(x))  # Normalize x[k] to [0, 1]
            axes1[l].plot(probabilities, results[0, :, k], marker='.', label=f"temp info = {x[k]:.2f}", linestyle=":",
                          color=color)
            axes1[l].set_title(transmat_mode)
            axes1[l].set_ylabel("L1")


        axes1[l].legend()


    fig1.suptitle("Pome " + omitter)
    fig1.supxlabel("Probability")
    fig1.savefig("Pome" + omitter + "Norm")



    if show:
        plt.show()

