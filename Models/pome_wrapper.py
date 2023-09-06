from pomegranate import *
import random
import numpy as np
from Models.model_wrapper import model_wrapper


class pome_wrapper(model_wrapper):
    def __init__(self, n_components, n_iter, emission_prob):
        super().__init__(n_components, n_iter)
        self.full_states = None
        self.states = None
        self.emission_prob = emission_prob

        # create model

        self._model = HiddenMarkovModel()
        self.add_states_with_dist()
        self.add_random_transitions()
        self._model.bake()

    def add_states_with_dist(self):
        dist_dict = self.create_dist_dict()
        states = []
        # creating states from dist and dict
        for state_dist in dist_dict:
            dist = DiscreteDistribution(state_dist)
            states.append(State(dist))
        # adding states
        self.states = states
        self.full_states = self.states + [self._model.start, self._model.end]
        self._model.add_states(self.states)


    def add_start_and_end_transitions(self):
        for state in self.states:
            self._model.add_transition(self._model.start, state, random.uniform(0, 1))
        for state in self.states:
            self._model.add_transition(state, self._model.end, random.uniform(0, 1))
        pass

    def add_random_transitions(self):
        self.add_start_and_end_transitions()
        for state_a in self.states:
            for state_b in self.states:
                self._model.add_transition(state_a, state_b, random.uniform(0, 1))

    def fit(self, data):
        # need to add the tags in here
        print("running fit")
        self._model.freeze_distributions()
        self._model.fit(sequences=data, max_iterations=1)

    def create_dist_dict(self):
        # Should create a dict of dist from the self.emission_prob, as seen in the example from ChatGPT:
        #     'distributions': [DiscreteDistribution({'A': 0.6, 'C': 0.1, 'G': 0.2, 'T': 0.1})
        #     , DiscreteDistribution({'A': 0.2, 'C': 0.3, 'G': 0.2, 'T': 0.3})],
        # for the init_params?
        state_probs = []
        for state_prob in self.emission_prob:
            state_probs.append(dict(enumerate(state_prob)))
        return state_probs

    @property
    def transmat(self):
        try:

            return self._model.dense_transition_matrix()
        except AttributeError as e:
            print(f"Model not initialized with fit, exception was raised: {e}")

    @property
    def startprob(self):
        try:
            return self._model.start
        except AttributeError as e:
            print(f"Model not initialized with fit, exception was raised: {e}")

    @property
    def emissionprob(self):
        try:
            return self.create_emissionprob()
        except AttributeError as e:
            print(f"Model not initialized with fit, exception was raised: {e}")

    def create_emissionprob(self):
        emissionprob = []
        for state in self._model.states:
            emissionprob.append(np.array(list(state.distribution.values())))
        return np.vstack(emissionprob)

