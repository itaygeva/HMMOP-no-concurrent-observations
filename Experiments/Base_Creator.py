from utils import *
import json


class Base_Creator:

    def __init__(self, config_path=None):
        self.config_path = "Config.JSON" if config_path is None else config_path
        self.default = None
        self.instances_type = None
        self.class_to_builder_dict = {}

    def _create_instance(self, instance_config):
        if instance_config["Class"] in self.class_to_builder_dict.keys():
            return self.class_to_builder_dict[instance_config["Class"]](instance_config)
        else:
            raise NotImplementedError(f'No class {instance_config["Class"]} implemented')

    def create_instances_dict(self):
        with open("config.json", "r") as json_file:
            config = json.load(json_file)
        instances_config = config[self.instances_type]
        self.default = config["default"]
        instances = {}
        for instance_config in instances_config:
            instances[instance_config["Name"]] = self._create_instance(instance_config)
        return instances
