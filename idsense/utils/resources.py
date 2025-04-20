from .logger import Logger

import requests

from tqdm import tqdm

from idsense import IDSENSE_DIR


def download_file(url, dest_path, chunk_size=8192):
    """Download a file from the given URL to the destination path."""

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("Content-Length", 0))
    with open(dest_path, "wb") as f, tqdm(
        total=total_size, unit="B", unit_scale=True, desc=dest_path.name
    ) as bar:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))


def download_resources(resources):
    """Download all the resources defined in the list."""

    for file_path, url in resources.items():
        dest_path = IDSENSE_DIR / file_path
        if not dest_path.exists():
            print(f"DOWNLOADING: {url}")
            download_file(url, dest_path)
        else:
            Logger.info(f"RESOURCE: {file_path} [ALREADY EXISTS]")
