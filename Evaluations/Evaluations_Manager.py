from utils import *
from Experiments.utils import *
from Experiments.Base_Creator import Base_Creator
import json

class Evaluation_Manager(Base_Creator):
    def __init__(self, models_dict,config_path=None):
        config_path = "Evaluations.JSON" if config_path is None else config_path
        super().__init__(config_path)
        self.instances_type = "evaluations"
        self.models_dict = models_dict
        self.key_name = "Test"
        self.name_to_test_dict = {
            "L1 Normalized": self._l1_normalized
        }

    def _l1_normalized(self, test_config):
        models = get_value_or_default("Models", test_config, self.default)
        if len(models) != 2:
            raise ValueError(f"L1 Normalized test expects 2 models, but got {len(models)}  instead")
        else:
            matrix1 = models[0].transmat
            matrix2 = models[1].transmat
            compare_mat_l1_norm(matrix1, matrix2)
            return True



