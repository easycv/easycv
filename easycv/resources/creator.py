import os
import yaml
import pathlib

from easycv.utils import running_on_notebook
from easycv.resources.download import download_file

# Import the correct tqdm
if running_on_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def create_resource(type='folder', show_progress=True):
    """
    Interactive tool to create resources. Asks user to input details about the resource and the \
    files contained inside. Creates the YAML file and adds it into the correct sources folder.

    :param show_progress: True to display progress bar, False otherwise, defaults to True
    :type show_progress: :class:`bool`, optional
    """

    if type not in ["folder", "package", "keyvalue"]:
        raise ValueError("Resource type doesn't exist! Valid types are folder, package and keyvalue")

    name = input("Resource name: ")

    if type == 'package':
        package_name = input("Package name: ")
        package_pip = input("Package name in pip: ")
        package_version = input("Package version: ")
        data = {"type": type, "package-name": package_name, "pip-name": package_pip, "version": package_version}
    elif type == 'keyvalue':
        n = int(input("Number of keys: "))
        keys = []
        for i in range(n):
            keys.append({"key": input("Key: "), "type": input("Type: ")})
        data = {"type": type, "keys": keys}
    else:
        n_files = input("Number of files: ")
        files = []

        for i in range(int(n_files)):
            filename = input("File {} name: ".format(i + 1))
            url = input("File {} url: ".format(i + 1))
            files.append({"filename": filename, "url": url})

        print("Downloading/hashing files...")

        files_iter = files
        if show_progress:
            files_iter = tqdm(
                files, bar_format="{percentage:3.0f}% {bar} {n_fmt}/{total_fmt}"
            )

        cwd = pathlib.Path(os.getcwd())

        for file in files_iter:
            sha256 = download_file(
                file["filename"], file["url"], cwd, show_progress=show_progress
            )
            file["sha256"] = sha256
            (cwd / file["filename"]).unlink()
        data = {"type": type, "files": files}


    filename = str(
        pathlib.Path(__file__).parent.absolute() / "sources" / (name.lower() + ".yaml")
    )
    with open(filename, "w") as file:
        yaml.dump(data, file)

    print("Resource created successfully!")


if __name__ == "__main__":
    create_resource()
