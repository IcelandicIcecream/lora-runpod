import os
import shutil
import mimetypes
import io
import re
import requests
from zipfile import ZipFile

from cog import Path

def clean_directory(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def clean_directories(paths):
    for path in paths:
        clean_directory(path)


def random_seed():
    return int.from_bytes(os.urandom(2), "big")

def extract_zip_and_flatten(zip_path, output_path):
    # extract zip contents, flattening any paths present within it
    if not os.path.exists(zip_path):
        r = requests.get(zip_path)
        uploaded_zip = ZipFile(io.BytesIO(r.content))
        read_zip = uploaded_zip
    else:
        read_zip = ZipFile(str(zip_path), "r")

    with read_zip as zip_ref:
        for zip_info in zip_ref.infolist():
            if zip_info.filename[-1] == "/" or zip_info.filename.startswith(
                "__MACOSX"
            ):
                continue
            mt = mimetypes.guess_type(zip_info.filename)
            if mt and mt[0] and mt[0].startswith("image/"):
                zip_info.filename = os.path.basename(zip_info.filename)
                zip_ref.extract(zip_info, output_path)


def get_output_filename(input_filename):
    temp_name = Path(input_filename).name
    return Path(re.sub("[^-a-zA-Z0-9_]", "", temp_name)).with_suffix(".safetensors")
