import Evaluations.utils as eval
from Experiments.utils import *
from Experiments.Base_Creator import Base_Creator
import json

class Evaluations_Manager(Base_Creator):
    def __init__(self, models_dict,config_path=None):
        config_path = "Evaluations.JSON" if config_path is None else config_path
        super().__init__(config_path, "evaluations")
        self.models_dict = models_dict
        self.key_name = "Test"
        self.class_to_builder_dict = {
            "L1 Normalized": self._l1_normalized
        }

    def _l1_normalized(self, test_config):
        models = get_value_or_default("Models", test_config, self.default)
        if len(models) != 2:
            raise ValueError(f"L1 Normalized test expects 2 models, but got {len(models)}  instead")
        else:
            matrix1 = self.models_dict[models[0]["Name"]].transmat
            matrix2 = self.models_dict[models[1]["Name"]].transmat
            eval.compare_mat_l1_norm(matrix1, matrix2)
            return True



