from Experiments.Creators.utils import *
import json
import pickle
import os
import inspect


class Base_Creator:
# TODO: Think whether to create a new class from builders.
# TODO: If every function gets the same config, why not have it as part of self.  Maybe not.
    def __init__(self, config_path=None, instances_type = ""):
        """
        :param config_path: the path to the JSON configuration file
        :param instances_type: the name of the type of instances to create
        (e.g. models, readers etc.)
        """
        self.config_path = "Config.JSON" if config_path is None else config_path
        self.default = None
        self.instances_type: str = instances_type
        self.class_to_builder_dict = {}  # maps out name to function
        self.key_name = "Class"  # the key name to map out the function

        # creates pickle variables
        parent_dir = os.path.dirname(inspect.getfile(Base_Creator))
        self.cache_dir = os.path.join(parent_dir, '../Cache', self.instances_type)
        os.makedirs(self.cache_dir, exist_ok=True)

    def _create_instance(self, instance_config):
        """
        returns an instance based on the instance config.
        only creates a new instance if Reinitialization is needed, or if a previous one doesn't exist.
        :param instance_config: the instance configuration loaded from the JSON file
        :return: an instance
        """
        if instance_config[self.key_name] in self.class_to_builder_dict.keys():
            if self._should_initialize(instance_config):
                # running the mapped function for creation of the instance
                instance = self.class_to_builder_dict[instance_config[self.key_name]](instance_config)
                # save the instance using pickle
                dump_initialized_class(self.cache_dir, instance_config, self.default, instance)
                return instance
            else:
                # load the instance using pickle
                return load_initialized_class(self.cache_dir, instance_config, self.default)
        else:
            raise NotImplementedError(f'{instance_config[self.key_name]} not implemented')

    def create_instances_dict(self):
        """
        creates a dictionary of instances (value) and their names (keys).
        The dictionary is created based on the JSON configuration file
        :return:
        """
        with open(self.config_path, "r") as json_file:
            config = json.load(json_file)
        instances_config = config[self.instances_type]
        # reading the default settings
        self.default = config["default"]

        instances = {}
        for instance_config in instances_config:
            instances[instance_config["Name"]] = self._create_instance(instance_config)
        return instances

    def _should_initialize(self, instance_config):
        """
        return a bool indicating whether to create a new instance
        :param instance_config: the instance configuration loaded from the JSON file
        :return: a bool indicating whether to create a new instance
        """
        if get_value_or_default("Reinitialize", instance_config, self.default):
            return True
        else:
            file_name = get_value_or_default("Name", instance_config, self.default) + ".pkl"
            cache_filename = os.path.join(self.cache_dir, file_name)
            if os.path.isfile(cache_filename):
                #  If the file exists, we do need to initialize it
                return False
            else:
                return True

