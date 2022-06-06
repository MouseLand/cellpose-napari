import os
import pathlib
from functools import partial

from napari.utils.translations import trans
from napari_plugin_engine import napari_hook_implementation

CELLPOSE_DATA = [
    ('rgb_3D.tif', trans._('Cells (3D+2Ch)')),
    ('rgb_2D.png', trans._('Cells 2D')),
]

def _load_cellpose_data(image_name, dname):
    # Import when users select one of sample data
    import numpy as np
    from cellpose.io import imread
    from cellpose.utils import download_url_to_file

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
    data = imread(cached_file)
    if '3D' in image_name:
        data = np.moveaxis(data, 0, 1)
    return [(data, {'name': dname})]

_DATA = {
    key: {'data': partial(_load_cellpose_data, key, dname), 'display_name': dname}
    for (key, dname) in CELLPOSE_DATA
}
globals().update({k: v['data'] for k, v in _DATA.items()})

@napari_hook_implementation
def napari_provide_sample_data():
    return _DATA
