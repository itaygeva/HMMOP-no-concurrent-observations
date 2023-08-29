## we can pretty much copy most of benny's code. However, we need to make sure that the format that we get is the same.
## Also, do we need the locations of the omissions?
## Also, benny's code works on 1-feature data. we want to work with n-feature data. We might need to make some changes there

import numpy as np

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
    relevant_obs = np.vstack(relevant_obs)

    if len(w) < 3:
        return sample_n_points_from_traj_binom(vec, pc)
    else:
        return relevant_obs, w


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

    all_relevant_observations = [ro[0] for ro in all_relevant_observations_and_ws]
    all_ws = [ro[1] for ro in all_relevant_observations_and_ws]

    return all_relevant_observations, all_ws