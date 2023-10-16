import os
import pickle


def get_value_or_default(var_name, config, default):
    if var_name in config.keys():
        return config[var_name]
    elif var_name in default.keys():
        return default[var_name]
    else:
        raise NotImplementedError(f"No default value set for {var_name}")


def load_initialized_class(cache_dir, config, default):
    file_name = get_value_or_default("Name", config, default) + ".pkl"
    cache_filename = os.path.join(cache_dir, file_name)
    with open(cache_filename, 'rb') as file:  # we already checked that the file exists
        instance = pickle.load(file)
    return instance


def dump_initialized_class(cache_dir, config, default, instance):
    file_name = get_value_or_default("Name", config, default) + ".pkl"
    cache_filename = os.path.join(cache_dir, file_name)
    with open(cache_filename, 'wb') as file:
        pickle.dump(instance, file)
