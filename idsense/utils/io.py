import json


def load_json(file_path):
    """Loads and returns the contents of a JSON file."""

    with open(file_path) as f:
        return json.load(f)
