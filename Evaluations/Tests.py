import matplotlib.pyplot as plt

from Experiments.Creators.utils import create_config_dataclass_objects as create_config
from Experiments.Creators.utils import create_and_fit_pipeline as create_pipeline
from Experiments.Creators.utils import load_or_initialize_pipeline as create_pipeline_from_configs
from Pipelines.pipeline import pipeline
from Pipelines.pome_pipeline import pome_pipeline
from Pipelines.matrix_pipeline import matrix_pipeline
import numpy as np
from Evaluations.utils import *
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
    readers = ["Synthetic 001"]
    num_of_iter = 5
    results = np.empty((2, 1, num_of_iter))
    fig, ax = plt.subplots(num_of_iter, 1, sharex=True, sharey=True, figsize=(10, 6))
    for j in range(1, num_of_iter + 1):
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
            #hmm_perm = find_optimal_permutation(pipeline_hmm_syn.means, pipeline_gt_syn.means)
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
        ax[j - 1].set_title(f"Iteration #{j}")
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


def compare_pipelines_vs_iter():
    readers = ["Synthetic 1", "Synthetic 01", "Synthetic 001"]
    reader = readers[-1]
    num_of_iter = 5
    results = np.empty((2, num_of_iter, 50))
    means_results = np.empty((2,num_of_iter, 50, 3))
    for j in range(1, num_of_iter + 1):
        pipeline_pome_syn: pipeline = create_pipeline(reader, "Pass All", "Pomegranate - Synthetic" + str(j))
        print(f"finished creating {pipeline_pome_syn}")
        """pipeline_hmm_syn: pipeline = create_pipeline(reader, "Pass All", "Hmmlearn" + str(j))
        print(f"finished creating {pipeline_hmm_syn}")"""
        pipeline_gibbs_syn: pipeline = create_pipeline(reader, "Pass All", "Gibbs Sampler" + str(j))
        print(f"finished creating {pipeline_gibbs_syn}")
        pipeline_gt_syn: pipeline = create_pipeline(reader, "Pass All", "Ground Truth" + str(j))
        print(f"finished creating {pipeline_gt_syn}")

        print("matrices compared l1 norm")
        pome_perm_list = find_optimal_permutation_list(pipeline_pome_syn.means_list, pipeline_gt_syn.means_list)
        #hmm_perm_list = find_optimal_permutation_list(pipeline_hmm_syn.means_list, pipeline_gt_syn.means_list)

        results[0, j - 1, :] = np.array(compare_mat_l1_norm_for_list(pipeline_gt_syn.transmat_list,
                                                                     reorient_matrix_list(
                                                                         pipeline_pome_syn.transmat_list,
                                                                         pome_perm_list)))
        means_results[0, j-1, :, :] = np.array(pipeline_pome_syn.means_list)
        """results[1, j - 1, :] = np.array(compare_mat_l1_norm_for_list(pipeline_gt_syn.transmat_list,
                                                                     reorient_matrix_list(
                                                                         pipeline_hmm_syn.transmat_list,
                                                                         hmm_perm_list)))"""
        results[1, j - 1, 0:19] = np.array(compare_mat_l1_norm_for_list(pipeline_gt_syn.transmat_list[0:19],
                                                                     pipeline_gibbs_syn.transmat_list))
        means_results[1, j - 1, 0:21, :] = np.array(pipeline_gibbs_syn.means_list)

        x = np.arange(1, 51)
        plt.figure(1)
        plt.plot(x, results[0, j - 1, :], marker='.', label=f"iter #{j}", linestyle=":")
        """plt.figure(2)
        plt.plot(x, results[1, j - 1, :], marker='D', label=f"iter #{j}", linestyle=":")"""
        plt.figure(2)
        plt.plot(x[0:19], results[1, j - 1, 0:19], marker='.', label=f"iter #{j}", linestyle=":")

        plt.figure(1+2*j)
        plt.plot(x, means_results[0, j - 1, :, :], marker='.', label=f"iter #{j}", linestyle="")
        plt.title(f"Pome iter #{j}")
        plt.axhline(y=0, color='r', linestyle=':')
        plt.axhline(y=10, color='r', linestyle=':')
        plt.axhline(y=20, color='r', linestyle=':')
        """plt.figure(2)
        plt.plot(x, results[1, j - 1, :], marker='D', label=f"iter #{j}", linestyle=":")"""
        plt.figure(2+2*j)
        plt.plot(x[0:21], means_results[1, j - 1, 0:21, :], marker='.', label=f"iter #{j}", linestyle="")
        plt.title(f"Gibbs iter #{j}")
        plt.axhline(y=0, color='r', linestyle=':')
        plt.axhline(y=10, color='r', linestyle=':')
        plt.axhline(y=20, color='r', linestyle=':')


    plt.figure(1)
    plt.title("Pome")
    plt.xlabel("iter #")
    plt.ylabel("L1 Norm")
    plt.legend()
    """plt.figure(2)
    plt.title("HMMLearn")
    plt.xlabel("iter #")
    plt.ylabel("L1 Norm")
    plt.legend()"""
    plt.figure(2)
    plt.title("Gibbs")
    plt.xlabel("iter #")
    plt.ylabel("L1 Norm")
    plt.legend()

    plt.show()





def compare_pipelines_for_different_bernoulli_prob():
    bernoulli_probabilities = np.linspace(0.2, 1, 5)
    results = np.empty((2, 5))
    for i, p in enumerate(bernoulli_probabilities):
        omitter_bernoulli_config: bernoulli_omitter_config
        reader_config, omitter_bernoulli_config, pipeline_pome_config = create_config(
            "Synthetic 001", "Bernoulli",
            "Pomegranate - Synthetic1")
        _, _, pipeline_gibbs_config = create_config("Synthetic 001", "Bernoulli",
                                                    "Gibbs Sampler1")
        _, omitter_pass_config, pipeline_gt_config = create_config("Synthetic 001", "Pass All",
                                                                   "Ground Truth1")

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
        pome_perm = find_optimal_permutation(pipeline_pome_syn.means, pipeline_gt_syn.means)
        gibbs_perm = find_optimal_permutation(pipeline_gibbs_syn.means, pipeline_gt_syn.means)

        results[0, i] = find_mat_diff(pipeline_gt_syn.transmat,
                                      reorient_matrix(pipeline_pome_syn.transmat, pome_perm)) / \
                        pipeline_gt_syn.transmat.shape[0]
        results[1, i] = find_mat_diff(pipeline_gt_syn.transmat, pipeline_gibbs_syn.transmat) / \
                        pipeline_gt_syn.transmat.shape[0]  # we don't change the orientation of the gibbs matrix

        print(pipeline_pome_syn.transmat)
        print(pipeline_gibbs_syn.transmat)
        print(pipeline_gt_syn.transmat)
        print(find_mat_diff(pipeline_gt_syn.transmat, pipeline_pome_syn.transmat) / \
              pipeline_gt_syn.transmat.shape[0])
        print(results[0, i])
        print(find_mat_diff(pipeline_gt_syn.transmat, pipeline_gibbs_syn.transmat) / \
              pipeline_gt_syn.transmat.shape[0])
        print(results[1, i])

    plt.figure(1)
    plt.plot(p, results[0], marker='o', label="pome")
    plt.plot(p, results[1], marker='o', label="gibbs")
    plt.ylabel("L1 normalized")
    plt.xlabel("probability of ")
    plt.legend()
    plt.show()
