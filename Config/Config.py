import os
from dataclasses import dataclass, field


def default_raw_data():
    data_dir = "../Data"
    return os.path.join(data_dir, 'Raw')


@dataclass
class Config:
    Name: str = field(default="Untitled", compare=False)
    Class: str = field(default="Undefined", repr=False)
    Reinitialize: bool = field(default=False, repr=False, compare=False)

@dataclass
class base_reader_config(Config):
    n_features: int = 1
    n_components: int = 1
    is_tagged: bool = False
    raw_dir: str = field(default_factory=default_raw_data, repr=False)


# region readers configs
@dataclass
class brown_corpus_reader_config(base_reader_config):
    path_to_data: str = field(default="", repr=False)
    path_to_tags: str = field(default="", repr=False)


@dataclass
class stocks_reader_config(base_reader_config):
    path_to_data: str = field(default="", repr=False)
    company: str = ""
    min_length: int = 0
    max_length: int = 0


@dataclass
class synthetic_reader_config(base_reader_config):
    n_samples: int = 0
    transmat: str = field(default="", repr=False)
    endprobs: str = field(default="", repr=False)
    startprobs: str = field(default="", repr=False)
    sigma: str = field(default="", repr=False)
    mues: str = field(default="", repr=False)
    params_dir: str = field(default="", repr=False)

@dataclass
class hmm_synthetic_reader_config(base_reader_config):
    n_samples: int = 0
    sentence_length: int = 0
    transmat: str = field(default="", repr=False)
    startprobs: str = field(default="", repr=False)
    sigma: str = field(default="", repr=False)
    mues: str = field(default="", repr=False)
    params_dir: str = field(default="", repr=False)

@dataclass
class my_synthetic_reader_config(base_reader_config):
    n_samples: int = 0
    sentence_length: int = 0
    transmat: str = field(default="", repr=False)
    startprobs: str = field(default="", repr=False)
    sigma: str = field(default="", repr=False)
    mues: str = field(default="", repr=False)
    params_dir: str = field(default="", repr=False)
    matrix_power: int = 0
    set_temporal: bool = False


# endregion

# region omitters configs
@dataclass
class base_omitter_config(Config):
    pass


@dataclass
class bernoulli_omitter_config(Config):
    prob_of_observation: float = 0.5
@dataclass
class geometric_omitter_config(Config):
    prob_of_observation: float = 0.5

@dataclass
class consecutive_bernoulli_omitter_config(Config):
    prob_of_observation: float = 0.5
@dataclass
class markov_chain_omitter_config(Config):
    prob_of_observation: float = 0.5

@dataclass
class uniform_skips_omitter_config(Config):
    n_skips: int = 4

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
