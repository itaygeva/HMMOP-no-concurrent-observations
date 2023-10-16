from utils import *
import json
import pickle
import os
import inspect


class Base_Creator:

    def __init__(self, config_path=None, instances_type = ""):
        self.config_path = "Config.JSON" if config_path is None else config_path
        self.default = None
        self.instances_type: str = instances_type
        self.class_to_builder_dict = {}
        self.key_name = "Class"

        # pickle variables
        parent_dir = os.path.dirname(inspect.getfile(Base_Creator))
        self.cache_dir = os.path.join(parent_dir, 'Cache', self.instances_type)
        os.makedirs(self.cache_dir, exist_ok=True)

    def _create_instance(self, instance_config):
        if instance_config[self.key_name] in self.class_to_builder_dict.keys():
            if self._should_initialize(instance_config):
                instance = self.class_to_builder_dict[instance_config[self.key_name]](instance_config)
                dump_initialized_class(self.cache_dir, instance_config, self.default, instance)
                return instance
            else:
                return load_initialized_class(self.cache_dir, instance_config, self.default)
        else:
            raise NotImplementedError(f'{instance_config[self.key_name]} not implemented')

    def create_instances_dict(self):
        with open(self.config_path, "r") as json_file:
            config = json.load(json_file)
        instances_config = config[self.instances_type]
        self.default = config["default"]
        instances = {}
        for instance_config in instances_config:
            instances[instance_config["Name"]] = self._create_instance(instance_config)
        return instances

    def _should_initialize(self, instance_config):
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

