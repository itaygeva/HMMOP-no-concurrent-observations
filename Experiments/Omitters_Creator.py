from Omission.omitters import base_omitter
from Omission.omitters import bernoulli_omitter
from Experiments.Base_Creator import Base_Creator
from utils import *
import json


class Omitters_Creator(Base_Creator):

    def __init__(self, config_path=None):
        super().__init__(config_path, instances_type = "omitters")
        self.class_to_builder_dict = {
            "base_omitter": self._build_base_omitter,
            "bernoulli_omitter": self._build_bernoulli_omitter
        }


    def _build_base_omitter(self, omitter_config):
        return base_omitter()


    def _build_bernoulli_omitter(self, omitter_config):
        prob_of_observation = get_value_or_default("Probability of Observation", omitter_config, self.default)
        return bernoulli_omitter(prob_of_observation)
