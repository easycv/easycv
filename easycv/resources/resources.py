import yaml

import shutil
from pathlib import Path

from easycv.resources.download import download_resource
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
    :return: A path to the resource folder or to the given file
    :rtype: :class:`pathlib.Path`
    """
    if resource_name in downloaded_resources():
        resource_folder = get_resources_folder() / resource_name
        if filename is None:
            return resource_folder
        else:
            file = resource_folder / filename
            if file.is_file():
                return file
            else:
                raise FileNotInResource(resource_name)
    elif resource_name in available_resources():
        print(resource_name + " is not installed. Downloading...")
        download_resource(resource_name, show_progress=show_progress)
        return get_resource(resource_name, filename=filename)
    else:
        raise InvalidResource(resource_name)


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
