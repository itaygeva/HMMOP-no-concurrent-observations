from Omission.utils import *

class base_omitter:
    def __init__(self):
        pass

    def omit(self, data):
        return data

class bernoulli_omitter(base_omitter):
    def __init__(self, prob_of_observation):
        self.prob_of_observation = prob_of_observation

# TODO: check if this still works with new data format.
# TODO: Also, make sure that all the readers are outputting the same format(torch or numpy?)
    def omit(self, data):
        return bernoulli_experiments(self.prob_of_observation, data)
