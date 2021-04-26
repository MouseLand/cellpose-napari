import pathlib
import os
from napari_plugin_engine import napari_hook_implementation
from napari.utils.translations import trans
from cellpose.utils import download_url_to_file
from cellpose.io import imread

CELLPOSE_DATA = [
    ('rgb_3D.tif', trans._('Cells (3D+2Ch)')),
    ('rgb_2D.png', trans._('Cells 2D')),
]

def _load_cellpose_data(image_name, dname):
    cp_dir = pathlib.Path.home().joinpath('.cellpose')
    cp_dir.mkdir(exist_ok=True)
    data_dir = cp_dir.joinpath('data')
    data_dir.mkdir(exist_ok=True)
    data_dir_2D = data_dir.joinpath('2D')
    data_dir_2D.mkdir(exist_ok=True)
    data_dir_3D = data_dir.joinpath('3D')
    data_dir_3D.mkdir(exist_ok=True)

    url = 'http://www.cellpose.org/static/data/' + image_name
    if '2D' in image_name:
        cached_file = str(data_dir_2D.joinpath(image_name))
    else:
        cached_file = str(data_dir_3D.joinpath(image_name))
    if not os.path.exists(cached_file):
        download_url_to_file(url, cached_file, progress=True)
    return [(imread(cached_file), {'name': dname})]

@napari_hook_implementation
def napari_provide_sample_data():
    from functools import partial
    return {
        key: {'data': partial(_load_cellpose_data, key, dname), 'display_name': dname}
        for (key, dname) in CELLPOSE_DATA
    }

    