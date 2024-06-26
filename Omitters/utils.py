## we can pretty much copy most of benny's code. However, we need to make sure that the format that we get is the same.
## Also, do we need the locations of the omissions?
## Also, benny's code works on 1-feature data. we want to work with n-feature data. We might need to make some changes there

import numpy as np
import random


## TODO: add verify short sentence function, which helps with the handling of short sentences.
# It needs to get the function for the recursion


def sample_n_points_from_traj_binom(vec, pc):
    """
    omits points along a trajectory using a binomial dist
    :param vec: the trajectory to preform omissions upon
    :param pc: the binomial probabilty
    :return: the omitted trajectory, and the omitted locations
    """
    relevant_obs = []
    w = []
    for i, obs in enumerate(vec):
        if np.random.rand() < pc:
            relevant_obs.append(obs)
            w.append(i)

    if len(vec) < 3:
        return np.array(vec), np.arange(len(vec))
    elif len(w) < 2:
        return sample_n_points_from_traj_binom(vec, pc)
    else:
        if vec.ndim == 1:
            return np.array(relevant_obs), w
        else:
            return np.vstack(relevant_obs), w


def bernoulli_experiments(p_prob_of_observation, all_full_sampled_trajs):
    """
    omits points along several trajectories using a binomial dist
    :param p_prob_of_observation: the binomial probability
    :param all_full_sampled_trajs: the trajectories to preform omissions upon
    :return: the omitted trajectories, and the omitted locations
    """
    all_relevant_observations_and_ws = []
    for vec in all_full_sampled_trajs:
        _new_vec = sample_n_points_from_traj_binom(vec, p_prob_of_observation)
        all_relevant_observations_and_ws.append(_new_vec)

    all_relevant_observations = [row[0] for row in all_relevant_observations_and_ws]
    all_ws = [row[1] for row in all_relevant_observations_and_ws]

    return all_relevant_observations, all_ws


def geometric_experiments(p_prob_of_observation, data):
    """
    omits points along several trajectories using a geometric distribution
    :param p_prob_of_observation: the success probability
    :param data: the trajectories to preform omissions upon
    :return: the omitted trajectories, and the omitted locations
    """
    ws = []
    omitted_data = []
    for sentence in data:
        omitted_sentence, w = geometric_omission(p_prob_of_observation, sentence)
        ws.append(w)
        omitted_data.append(omitted_sentence)
    return omitted_data, ws


def geometric_omission(p_prob_of_observation, sentence):
    """
    omits points along a trajectory by sampling each time the jump between seen emissions from a geometric dist.
    :param p_prob_of_observation: the success probability of the geometric dist
    :param sentence: the trajectory to preform omissions upon
    :return: the omitted trajectory, and the omitted locations
    """
    skips = np.random.geometric(p=p_prob_of_observation, size=len(sentence))
    skips[0] -= 2  # this is so we can start at 0
    skips += np.ones_like(skips)
    indexes = np.cumsum(skips)  # this is to get the indexes

    w = indexes[indexes < len(sentence)]
    omitted_sentence = sentence[w]

    if len(sentence) < 3:
        return np.array(sentence), np.arange(len(sentence))
    elif len(w) < 2:
        return geometric_omission(p_prob_of_observation, sentence)
    else:
        if sentence.ndim == 1:
            return np.array(omitted_sentence), w
        else:
            return np.vstack(omitted_sentence), w


def consecutive_bernoulli_experiments(p_prob_of_observation, data):
    """
    omits consecutive points along several trajectories using a binomial dist
    :param p_prob_of_observation: the binomial probability
    :param data: the trajectories to preform omissions upon
    :return: the omitted trajectories, and the omitted locations
    """
    ws = []
    omitted_data = []
    for sentence in data:
        omitted_sentence, w = consecutive_bernoulli_omission(p_prob_of_observation, sentence)
        ws.append(w)
        omitted_data.append(omitted_sentence)
    return omitted_data, ws


def consecutive_bernoulli_omission(p_prob_of_observation, sentence):
    """
    omits points along a trajectory by choosing to skip the next point using a bernoulli trial.
    The larger the probability the more consecutive emissions we will see.
    :param p_prob_of_observation: the bernoulli probability
    :param sentence: the trajectory to preform omissions upon
    :return: the omitted trajectory, and the omitted locations
    """

    steps = np.random.choice([1, 2], p=[p_prob_of_observation, 1 - p_prob_of_observation], size=len(sentence))
    steps[0] -= 1  # this is so we can start at 0
    indexes = np.cumsum(steps)  # this is to get the indexes

    w = indexes[indexes < len(sentence)]
    omitted_sentence = sentence[w]

    if len(sentence) < 3:
        return np.array(sentence), np.arange(len(sentence))
    elif len(w) < 2:
        return consecutive_bernoulli_omission(p_prob_of_observation, sentence)
    else:
        if sentence.ndim == 1:
            return np.array(omitted_sentence), w
        else:
            return np.vstack(omitted_sentence), w


def markov_chain_experiments(epsilon, data):
    """
    omits points along several trajectories using a markov chain
    :param epsilon: the epsilon to shift the balance of the emit and omit states
    :param data: the trajectories to preform omissions upon
    :return: the omitted trajectories, and the omitted locations
    """
    ws = []
    omitted_data = []
    for sentence in data:
        omitted_sentence, w = markov_chain_omission(epsilon, sentence)
        ws.append(w)
        omitted_data.append(omitted_sentence)
    return omitted_data, ws


def markov_chain_omission(epsilon, sentence):
    """
    Samples a sentence from a 2 state (emit, omit) markov chain. Omits points along a trajectory using the corresponding states.
    If epsilon is 0, we will have a balanced transition matrix and starting probability. meaning all values are half.
    Epsilon shift the balance to one of the states. The larger epsilon is the more dominant 'emit' will be.
    The smaller it is the more dominant 'omit' will be.
    :param epsilon: the shift from balanced transition matrix and starting probability
    :param sentence: the trajectory to preform omissions upon
    :return: the omitted trajectory, and the omitted locations
    """
    transmat = np.array([[0.5 - epsilon, 0.5 + epsilon], [0.5 - epsilon, 0.5 + epsilon]])
    state = np.random.choice([0, 1], p=[0.5 - epsilon, 0.5 + epsilon])
    w = []
    for i in range(len(sentence)):
        if state:
            w.append(i)
        state = np.random.choice([0, 1], p=transmat[state])
    omitted_sentence = sentence[w]

    if len(sentence) < 3:
        return np.array(sentence), np.arange(len(sentence))
    elif len(w) < 2:
        return markov_chain_omission(epsilon, sentence)
    else:
        if sentence.ndim == 1:
            return np.array(omitted_sentence), w
        else:
            return np.vstack(omitted_sentence), w


def uniform_skips_experiment(num_of_skips, data):
    """
    omits points along several trajectories by jumping between seen emission using a uniform dist to determine the jump's size
    :param num_of_skips: the maximum number of omissions between 2 seen emissions
    :param data: the trajectories to preform omissions upon
    :return: the omitted trajectories, and the omitted locations
    """
    ws = []
    omitted_data = []
    for sentence in data:
        omitted_sentence, w = uniform_skips_omission(num_of_skips, sentence)
        ws.append(w)
        omitted_data.append(omitted_sentence)
    return omitted_data, ws


def uniform_skips_omission(num_of_skips, sentence):
    """
    omits points along a trajectory by sampling each time the jump between seen emissions from a uniform dist.
    :param num_of_skips: the maximum size of a single skip
    :param sentence: the trajectory to preform omissions upon
    :return: the omitted trajectory, and the omitted locations
    """
    skips = np.random.randint(low=1, high=num_of_skips + 1, size=sentence.shape)
    skips[0] -= 2  # this is so we can start at 0
    skips += np.ones_like(skips)
    indexes = np.cumsum(skips)  # this is to get the indexes

    w = indexes[indexes < len(sentence)]
    omitted_sentence = sentence[w]

    if len(sentence) < 3:
        return np.array(sentence), np.arange(len(sentence))
    elif len(w) < 2:
        return geometric_omission(num_of_skips, sentence)
    else:
        if sentence.ndim == 1:
            return np.array(omitted_sentence), w
        else:
            return np.vstack(omitted_sentence), w
