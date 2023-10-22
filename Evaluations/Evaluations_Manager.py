import Evaluations.utils as eval
from Experiments.Creators.utils import *
from Experiments.Creators.Base_Creator import Base_Creator


class Evaluations_Manager(Base_Creator):
    def __init__(self, models_dict,config_path=None):
        """
        :param models_dict: the dictionary containing the models' instances
        :param config_path: the path to the JSON configuration file.
        """
        config_path = "Evaluations.JSON" if config_path is None else config_path
        super().__init__(config_path, "evaluations")
        self.models_dict = models_dict
        self.key_name = "Test"
        self.class_to_builder_dict = {
            "L1 Normalized": self._l1_normalized
        }

    def _l1_normalized(self, test_config):
        """
        prints the normalized l1 distance between the matrices
        :param test_config: the evaluator configuration loaded from the JSON file
        :return: whether the test has run successfully
        """
        models = get_value_or_default("Models", test_config, self.default)
        if len(models) != 2:
            # TODO: return false instead of exception
            raise ValueError(f"L1 Normalized test expects 2 models, but got {len(models)}  instead")
        else:
            matrix1 = self.models_dict[models[0]["Name"]].transmat  # the transmat of the first model
            matrix2 = self.models_dict[models[1]["Name"]].transmat  # the transmat of the first model
            eval.compare_mat_l1_norm(matrix1, matrix2)
            return True



