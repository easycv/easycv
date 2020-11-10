import sys
import yaml
import subprocess

import shutil
from pathlib import Path

import easycv.resources.download
from easycv.errors import InvalidResource, FileNotInResource


def available_resources():
    """
    Obtains a list with the names of all available resources. An available resource is a resource \
    that can be downloaded. A resource that was already downloaded will still be displayed here.

    :return: List containing the names of all downloaded available
    :rtype: :class:`list`
    """
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


def downloaded_resources():
    """
    Obtains a list with the names of all downloaded resources.

    :return: List containing the names of all downloaded resources
    :rtype: :class:`list`
    """
    downloaded = []
    resources_folder = get_resources_folder()

    for x in resources_folder.iterdir():
        if x.is_dir() and x.name in available_resources():
            downloaded.append(x.name)

    return downloaded


def get_resource(resource_name, filename=None, show_progress=True):
    """
    Obtains the path to a resource. If the resource is not downloaded, it is downloaded first.\
    Supports progress display.

    :param resource_name: Name of the desired resource
    :type resource_name: :class:`str`
    :param filename: Desired file, defaults to None
    :type filename: :class:`str`, optional
    :param show_progress: True to display a progress bar, False otherwise, defaults to True
    :type show_progress: :class:`bool`, optional
    """
    if resource_name in downloaded_resources():
        info = load_resource_info(resource_name)
        resource_type = info["type"]

        resource_folder = get_resources_folder() / resource_name
        if resource_type == "folder":
            if filename is None:
                return resource_folder
            else:
                file = resource_folder / filename
                if file.is_file():
                    return file
                else:
                    raise FileNotInResource(resource_name)
        elif resource_type == "keyvalue":
            return read_key_values(resource_name)
        elif resource_type == "package":
            return __import__(info["package-name"])
    elif resource_name in available_resources():
        info = load_resource_info(resource_name)
        resource_type = info["type"]
        if resource_type == "folder":
            print(resource_name + " is not installed. Downloading...")
            easycv.resources.download.download_resource(
                resource_name, show_progress=show_progress
            )
        elif resource_type == "keyvalue":
            print(
                resource_name
                + " doesn't have any values stored yet. Please insert the required data:"
            )
            store_key_values(resource_name, info)
        elif resource_type == "package":
            print(
                "Package " + info["package-name"] + " is not installed. Downloading..."
            )
            install_package(resource_name, info)
        return get_resource(resource_name, filename=filename)
    else:
        raise InvalidResource(resource_name)


def install_package(resource_name, info):
    try:
        if "version" in info:
            info["pip-name"] += "==" + info["version"]
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", info["pip-name"]]
        )
    except Exception:
        raise OSError("Problem installing package!")

    resource_folder = get_resources_folder() / resource_name

    if resource_folder.is_dir():
        raise ValueError("Already exists!")

    resource_folder.mkdir()


def read_key_values(resource_name):
    resource_folder = get_resources_folder() / resource_name
    yaml_file = str(resource_folder / "database.yaml")

    with open(yaml_file, "r") as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)


def store_key_values(resource_name, info):
    resource_folder = get_resources_folder() / resource_name

    if resource_folder.is_dir():
        raise ValueError("Already exists!")

    resource_folder.mkdir()
    yaml_file = str(resource_folder / "database.yaml")

    try:
        d = {}
        for key in info["keys"]:
            d[key["key"]] = input(key["key"] + ":  ")

        with open(yaml_file, "w") as file:
            yaml.dump(d, file)
    except Exception as e:  # Clear folder if something went wrong
        print("Abort: " + str(e))
        print("Cleaning up...")
        shutil.rmtree(resource_folder)


def delete_resource(resource_name):
    """
    Deletes a local resource. If the resource is not downloaded nothing is executed.

    :param resource_name: Name of the resource to delete
    :type resource_name: :class:`str`
    """
    if resource_name in downloaded_resources():
        resource_folder = get_resources_folder() / resource_name
        shutil.rmtree(resource_folder)


def get_resources_folder():
    """
    Obtains a path to the resources folder.

    :return: Path to the local resources folder
    :rtype: :class:`pathlib.Path`
    """
    folder = Path(__file__).parent.absolute() / "downloaded"
    if not folder.is_dir():
        folder.mkdir()
    return folder
