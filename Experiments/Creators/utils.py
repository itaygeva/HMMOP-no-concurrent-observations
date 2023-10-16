import os
import pickle


def get_value_or_default(var_name, config, default):
    """
    returns the value of the var from the instance's config dictionary.
    If it doesn't find it there, returns the value in the default config dictionary.
    :param var_name: the name of the variable (the key in the JSON)
    :param config: the instance configuration loaded from the JSON file
    :param default: the default configuration loaded from the JSON file
    :return: the found value of the var
    """
    if var_name in config.keys():
        return config[var_name]
    elif var_name in default.keys():
        return default[var_name]
    else:
        raise NotImplementedError(f"No default value set for {var_name}")


def load_initialized_class(cache_dir, config, default):
    """
    returns an instance loaded from a saved pickle file
    :param cache_dir: the directory of the saved pickle files
    :param config: the instance configuration loaded from the JSON file
    :param default: the default configuration loaded from the JSON file
    :return: a loaded instance
    """
    file_name = get_value_or_default("Name", config, default) + ".pkl"
    cache_filename = os.path.join(cache_dir, file_name)
    with open(cache_filename, 'rb') as file:  # we already checked that the file exists
        instance = pickle.load(file)
    return instance


def dump_initialized_class(cache_dir, config, default, instance):
    """
    saves the given instance to a pickle file.
    :param cache_dir: the directory in which to pickle the instance.
    :param config: the instance configuration loaded from the JSON file
    :param default: the default configuration loaded from the JSON file
    :param instance: the instance to save
    """
    file_name = get_value_or_default("Name", config, default) + ".pkl"
    cache_filename = os.path.join(cache_dir, file_name)
    with open(cache_filename, 'wb') as file:
        pickle.dump(instance, file)
