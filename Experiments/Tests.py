import matplotlib.pyplot as plt
import numpy as np

import Config.Config
from Experiments.utils import create_config_dataclass_objects as create_config
from Experiments.utils import create_and_fit_pipeline as create_pipeline
from Experiments.utils import load_or_initialize_pipeline as create_pipeline_from_configs
from Pipelines.pipeline import pipeline
from Experiments.evalUtils import *
from Config.Config import *


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
                results[0, j, i, k - 1] = compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_pome_syn.transmat)
                results[1, j, i, k - 1] = compare_mat_l1_norm(pipeline_gt_syn.transmat, pipeline_gibbs_syn.transmat)
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
