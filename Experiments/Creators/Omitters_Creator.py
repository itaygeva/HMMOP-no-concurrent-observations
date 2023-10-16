from Omitters.omitters import base_omitter
from Omitters.omitters import bernoulli_omitter
from Experiments.Creators.Base_Creator import Base_Creator
from Experiments.Creators.utils import *


class Omitters_Creator(Base_Creator):

    def __init__(self, config_path=None):
        """
        :param config_path: the path to the JSON configuration file
        """
        super().__init__(config_path, instances_type = "omitters")
        self.class_to_builder_dict = {
            "base_omitter": self._build_base_omitter,
            "bernoulli_omitter": self._build_bernoulli_omitter
        }


    def _build_base_omitter(self, omitter_config):
        """
        creates a base omitter (passed all the data)
        :param omitter_config: the omitter configuration loaded from the JSON file
        :return: a base omitter
        """
        return base_omitter()


    def _build_bernoulli_omitter(self, omitter_config):
        """
        creates a bernoulli omitter
        :param omitter_config: the omitter configuration loaded from the JSON file
        :return: a bernoulli omitter
        """
        prob_of_observation = get_value_or_default("Probability of Observation", omitter_config, self.default)
        return bernoulli_omitter(prob_of_observation)
