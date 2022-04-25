import json
import os


class Config:
    def __init__(self, secret_folder: str):
        """
        Looks for .json files inside `secret_folder` and adds it as dict properties
        """
        for name in os.listdir(secret_folder):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(secret_folder, name), "r") as f:
                name = name.split(".")[0]
                self.__dict__[name] = json.load(f)
