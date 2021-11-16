import json


def load_dataset(path):
    with open(path) as f:
        data = json.load(f)
    return data
