from dataclasses import dataclass, fields, field
from typing import List
import json


@dataclass
class Config:
    Name: str = field(default="Untitled", compare=False)
    Class: str = field(default="Undefined", repr=False)
    Reinitialize: bool = field(default=False, repr=False, compare=False)


# region readers configs
@dataclass
class brown_corpus_reader_config(Config):
    Class: str = field(default="brown_corpus_reader", repr=False)
    Path_to_Data: str = ""


@dataclass
class stocks_reader_config(Config):
    Class: str = field(default="stocks_reader", repr=False)
    Path_to_Data: str = ""


@dataclass
class synthetic_reader_config(Config):
    Class: str = field(default="synthetic_reader", repr=False)
    Path_to_Data: str = ""
    n_components: int = 0
    n_samples: int = 0
    args: List[int] = field(default_factory=list)


# endregion

# region omitters configs
@dataclass
class base_omitter_config(Config):
    pass

@dataclass
class bernoulli_omitter_config(Config):
    prob_of_observation: float = 0.5


# endregion

# region models configs
@dataclass
class pipeline_config(Config):
    n_components: int = 0
    n_iter: int = 0

@dataclass
class gibbs_sampler_pipeline_config(pipeline_config):
    pass
@dataclass
class hmmlearn_pipeline_config(pipeline_config):
    distribution: int = ""


@dataclass
class matrix_pipeline_config(pipeline_config):
    pass


@dataclass
class pome_pipeline_config(pipeline_config):
    distribution: int = ""
    n_features: int = 0
    freeze_distributions: bool = False

# endregion
