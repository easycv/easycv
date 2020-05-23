import yaml

import shutil
from pathlib import Path

from easycv.errors import InvalidResource
from easycv.resources.download import download_model


def get_resources_folder():
    return Path(__file__).parent.absolute() / "downloaded"


def downloaded_resources():
    downloaded = []
    model_folder = Path(__file__).parent.absolute() / "downloaded"

    for x in model_folder.iterdir():
        if x.is_dir() and x.name in available_resources():
            downloaded.append(x.name)

    return downloaded


def available_resources():
    sources = Path(__file__).parent.absolute() / "sources"
    return [x.stem for x in sources.glob("*.yaml") if x.is_file()]


def load_resource_info(model_name):
    sources = Path(__file__).parent.absolute() / "sources"
    model_source = sources / (model_name + ".yaml")
    with open(model_source, "r") as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)


def get_resource(resource_name, show_progress=False):
    if resource_name in downloaded_resources():
        return get_resources_folder() / resource_name
    elif resource_name in available_resources():
        print(resource_name + " is not installed. Downloading...")
        download_model(resource_name, show_progress=show_progress)
        get_resource(resource_name)
    else:
        raise InvalidResource(resource_name)


def delete_resource(resource_name):
    if resource_name in downloaded_resources():
        resource_folder = get_resources_folder() / resource_name
        shutil.rmtree(resource_folder)
