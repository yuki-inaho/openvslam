import yaml


def read_yaml(yaml_path_str: str):
    with open(yaml_path_str, "r") as f:
        yaml_data = yaml.load(f)
    return yaml_data