from Models.pome_wrapper import pome_wrapper
from Models.gibbs_sampler_wrapper import gibbs_sampler_wrapper
from Models.matrix_wrapper import matrix_wrapper

from Experiments.Creators.Base_Creator import Base_Creator
from Experiments.Creators.utils import *


class Models_Creator(Base_Creator):
    def __init__(self, readers_dict, omitters_dict, config_path=None):
        super().__init__(config_path, "models")
        self.readers_dict = readers_dict
        self.omitters_dict = omitters_dict
        self.class_to_builder_dict = {
            "pome_wrapper": self._build_pome_wrapper,
            "gibbs_sampler_wrapper": self._build_gibbs_sampler_wrapper,
            "matrix_wrapper": self._build_matrix_wrapper
        }

    def _create_reader_and_omitter(self, model_config):
        reader_name = get_value_or_default("Reader", model_config, self.default)
        omitter_name = get_value_or_default("Omitter", model_config, self.default)
        reader = self.readers_dict[reader_name]
        omitter = self.omitters_dict[omitter_name]
        return reader, omitter

    def _create_omitted_data(self, model_config):
        reader, omitter = self._create_reader_and_omitter(model_config)
        data = reader.get_obs()
        return omitter.omit(data)

    def _create_omitted_data_and_emission_prob(self, model_config):
        reader, omitter = self._create_reader_and_omitter(model_config)
        data = reader.get_obs()
        emission_prob = reader.get_emission_prob()
        return omitter.omit(data), emission_prob

    def _build_gibbs_sampler_wrapper(self, model_config):
        omitted_data = self._create_omitted_data(model_config)
        n_states = get_value_or_default("Number of States", model_config, self.default)
        n_iter = get_value_or_default("Number of Iterations", model_config, self.default)
        model = gibbs_sampler_wrapper(n_states, n_iter)
        model.fit(omitted_data)
        return model

    def _build_pome_wrapper(self, model_config):
        # reading config values
        n_states = get_value_or_default("Number of States", model_config, self.default)
        n_iter = get_value_or_default("Number of Iterations", model_config, self.default)
        distribution_type = get_value_or_default("Distribution Type", model_config, self.default)
        freeze_distributions = get_value_or_default("Freeze Distributions", model_config, self.default)
        # TODO: get rid of n_features in init of pome_wrapper
        n_features = get_value_or_default("Number of Features", model_config, self.default)

        pass_emission_prob = get_value_or_default("Pass Emission Probabilities", model_config, self.default)
        if pass_emission_prob:  # passing emission probability matrix
            omitted_data, emission_prob = self._create_omitted_data_and_emission_prob(model_config)
            model = pome_wrapper(n_states, n_iter, distribution_type, n_features, emission_prob, freeze_distributions)
        else:  # not passing emission probability matrix
            omitted_data = self._create_omitted_data(model_config)
            model = pome_wrapper(n_states, n_iter, distribution_type, n_features)

        model.fit(omitted_data)
        return model

    def _build_matrix_wrapper(self, model_config):
        reader_name = get_value_or_default("Reader", model_config, self.default)
        reader = self.readers_dict[reader_name]
        model = matrix_wrapper()
        model.fit(reader.transition_mat)  # TODO: fix this to be normal (like get_transition_mat or something)
        return model
