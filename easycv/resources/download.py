import urllib.error as url_errors
from urllib.request import urlopen

import shutil
import hashlib

import easycv.resources.resources as resources
from easycv.utils import running_on_notebook
from easycv.errors import ErrorDownloadingResource

# Import the correct tqdm
if running_on_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def download_file(file, folder, chunk_size=8192, show_progress=False):
    file_hash = hashlib.sha256()
    filename = folder / file["filename"]

    try:
        response = urlopen(file["url"])
        size = int(response.info().get("Content-Length").strip())
        chunk = min(size, chunk_size)

        progress_bar = downloaded_size = None

        if show_progress:
            downloaded_size = 0
            progress_bar = tqdm(
                total=100,
                desc=file["filename"],
                bar_format="{percentage:3.0f}% {bar} {desc}",
                leave=False,
            )

        with open(filename, "wb") as local_file:
            while True:
                data_chunk = response.read(chunk)
                if not data_chunk:
                    break

                local_file.write(data_chunk)
                file_hash.update(data_chunk)

                if show_progress:
                    downloaded_size += len(data_chunk)
                    progress_bar.n = int(100 * downloaded_size / size)
                    progress_bar.refresh()

        if file["sha256"] != file_hash.hexdigest():
            raise RuntimeError("File hashes don't match")

    except (url_errors.HTTPError, url_errors.URLError) as e:
        raise ErrorDownloadingResource(e.reason)


def download_resource(resource_name, show_progress=False):
    info = resources.load_resource_info(resource_name)
    files = info["files"]

    resources_folder = resources.get_resources_folder()
    resource_folder = resources_folder / resource_name

    if resource_folder.is_dir():
        raise ValueError("Already downloaded")

    resource_folder.mkdir()

    try:
        if show_progress:
            for file in tqdm(
                files, bar_format="{percentage:3.0f}% {bar} {n_fmt}/{total_fmt}"
            ):
                download_file(file, resource_folder, show_progress=True)
        else:
            for file in files:
                download_file(file, resource_folder)

    except Exception as e:  # Clear folder if something went wrong
        print("Abort: " + str(e))
        print("Cleaning up...")
        shutil.rmtree(resource_folder)
