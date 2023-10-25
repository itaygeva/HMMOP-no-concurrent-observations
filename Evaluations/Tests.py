from Experiments.Creators.utils import create_config_dataclass_objects as create_config
from Experiments.Creators.utils import create_and_fit_pipeline as create_pipeline
from Experiments.Creators.utils import load_or_initialize_pipeline as create_pipeline_from_configs
from Pipelines.pome_pipeline import pome_pipeline
from Pipelines.matrix_pipeline import matrix_pipeline
import numpy as np

def l1_vs_p():
    ## TODO: Create default for readers, omitters, models
    ## TODO: Create dataclasses for configurations, as seen in playground.
    # Create dict of attributes from the dataclass, and fill it up using json and default.
    # Then create the dataclass and pass to the object
    # and then the model etc will hold a config obj
    pipeline_pome: pome_pipeline = create_pipeline("Synthetic", "Pass All", "Pomegranate")
    pipeline_ground_truth: matrix_pipeline = create_pipeline("Synthetic", "Pass All", "Ground Truth")

    print(np.array([transmat.numpy() for transmat in pipeline_pome.transmat_list]))


