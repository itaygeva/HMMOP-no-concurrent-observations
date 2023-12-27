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
    :param p_prob_of_observation: the binomial probabilty
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
    ws = []
    omitted_data = []
    for sentence in data:
        omitted_sentence, w = geometric_omission(p_prob_of_observation, sentence)
        ws.append(w)
        omitted_data.append(omitted_sentence)
    return omitted_data, ws


def geometric_omission(p_prob_of_observation, sentence):
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
    ws = []
    omitted_data = []
    for sentence in data:
        omitted_sentence, w = consecutive_bernoulli_omission(p_prob_of_observation, sentence)
        ws.append(w)
        omitted_data.append(omitted_sentence)
    return omitted_data, ws


def consecutive_bernoulli_omission(p_prob_of_observation, sentence):
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
    ws = []
    omitted_data = []
    for sentence in data:
        omitted_sentence, w = markov_chain_omission(epsilon, sentence)
        ws.append(w)
        omitted_data.append(omitted_sentence)
    return omitted_data, ws


def markov_chain_omission(epsilon, sentence):
    transmat = np.array([[0.5 - epsilon, 0.5 + epsilon], [0.5 + epsilon, 0.5 - epsilon]])
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
