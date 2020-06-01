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


def download_file(filename, url, path, chunk_size=8192, show_progress=False):
    file_hash = hashlib.sha256()
    try:
        response = urlopen(url)
        size = int(response.info().get("Content-Length").strip())
        chunk = min(size, chunk_size)

        progress_bar = downloaded_size = None

        if show_progress:
            downloaded_size = 0
            progress_bar = tqdm(
                total=100,
                desc=filename,
                bar_format="{percentage:3.0f}% {bar} {desc}",
                leave=False,
            )

        with open(path / filename, "wb") as local_file:
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

    except (url_errors.HTTPError, url_errors.URLError) as e:
        raise ErrorDownloadingResource(e.reason)

    return file_hash.hexdigest()


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
            files = tqdm(
                files, bar_format="{percentage:3.0f}% {bar} {n_fmt}/{total_fmt}"
            )

        for file in files:
            sha256 = download_file(
                file["filename"],
                file["url"],
                resource_folder,
                show_progress=show_progress,
            )
            if file["sha256"] != sha256:
                raise RuntimeError("File hashes don't match")

    except Exception as e:  # Clear folder if something went wrong
        print("Abort: " + str(e))
        print("Cleaning up...")
        shutil.rmtree(resource_folder)
