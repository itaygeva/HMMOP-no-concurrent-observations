import os
import pickle
from Experiments.settings import CONFIG_PATH, CACHE_DIR
import Experiments.experiment
import inspect
import json
from dataclasses import dataclass, fields, field
from Config.Config import *
from Data.Readers import *
from Omitters import *
from Pipelines import *

## TODO: See if we can import in runtime

def get_value_or_default(var_name, config, default):
    if var_name in config.keys():
        return config[var_name]
    elif var_name in default.keys():
        return default[var_name]

    else:
        raise NotImplementedError(f"No default value set for {var_name} in config: {config}, with default: {default}")


def load_initialized_class(file_path):
    with open(file_path, 'rb') as file:  # we already checked that the file exists
        instance = pickle.load(file)
    return instance


def dump_initialized_class(file_path, instance):
    file_directory = os.path.dirname(file_path)
    os.makedirs(file_directory, exist_ok=True)
    with open(file_path, 'wb') as file:
        pickle.dump(instance, file)


def get_configs_from_json(reader_name, omitter_name, model_name):
    config_path = CONFIG_PATH
    with open(config_path, "r") as json_file:
        config = json.load(json_file)
    reader_config = [reader_config for reader_config in config["readers"] if reader_config["Name"] == reader_name][0]
    omitter_config = \
        [omitter_config for omitter_config in config["omitters"] if omitter_config["Name"] == omitter_name][0]
    model_config = [model_config for model_config in config["models"] if model_config["Name"] == model_name][0]
    default_config = config["default"]
    return reader_config, omitter_config, model_config, default_config


def create_config(instance_config, default_config):
    fields_list = fields(eval(instance_config["Class"] + "_config"))
    fields_names = [field_inst.name for field_inst in fields_list]
    config = {}
    for field_name in fields_names:
        config[field_name] = get_value_or_default(field_name, instance_config, default_config)
    initialized_config = eval(config["Class"] + "_config")(**config)
    return initialized_config


def create_config_dataclass_objects(reader_name, omitter_name, model_name):
    reader_config, omitter_config, model_config, default_config = get_configs_from_json(reader_name, omitter_name,
                                                                                        model_name)

    reader_config_obj = create_config(reader_config, default_config)
    omitter_config_obj = create_config(omitter_config, default_config)
    model_config_obj = create_config(model_config, default_config)

    return reader_config_obj, omitter_config_obj, model_config_obj


def load_or_initialize_pipeline(reader_config, omitter_config, model_config):
    parent_dir = os.path.dirname(inspect.getfile(Experiments.experiment))
    cache_dir = os.path.join(parent_dir, 'Cache')
    readers_dir = os.path.join(cache_dir, "readers")
    omitters_dir = os.path.join(cache_dir, 'omitters')
    pipelines_dir = os.path.join(cache_dir, 'pipelines')

    reader_file_path = os.path.join(readers_dir, str(reader_config))
    omitter_file_path = os.path.join(omitters_dir, str(omitter_config))
    pipeline_file_path = os.path.join(pipelines_dir,
                                      str(model_config), str(omitter_config), str(reader_config))

    # region loading or initializing reader
    reader_should_initialize = (not (os.path.exists(reader_file_path))) or reader_config.Reinitialize
    if reader_should_initialize:
        reader = eval(reader_config.Class)(reader_config)
        dump_initialized_class(reader_file_path, reader)
    else:
        reader = load_initialized_class(reader_file_path)
    # endregion

    # region loading or initializing omitter
    omitter_should_initialize = (not (os.path.exists(omitter_file_path))) or omitter_config.Reinitialize
    if omitter_should_initialize:
        omitter = eval(omitter_config.Class)(omitter_config)
        dump_initialized_class(omitter_file_path, omitter)
    else:
        omitter = load_initialized_class(omitter_file_path)
    # endregion

    # region loading or initializing pipeline
    pipeline_should_initialize = (not (os.path.exists(pipeline_file_path))) or model_config.Reinitialize \
                                 or omitter_should_initialize or reader_should_initialize
    if pipeline_should_initialize:
        pipeline = eval(model_config.Class)(reader, omitter, model_config)
        pipeline.fit()
        dump_initialized_class(pipeline_file_path, pipeline)
    else:
        pipeline = load_initialized_class(pipeline_file_path)
    # endregion

    return pipeline


def create_and_fit_pipeline(reader_name, omitter_name, model_name):
    reader_config, omitter_config, model_config = create_config_dataclass_objects(reader_name, omitter_name, model_name)
    pipeline = load_or_initialize_pipeline(reader_config, omitter_config, model_config)
    return pipeline
