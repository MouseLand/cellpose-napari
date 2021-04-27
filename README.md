# cellpose-napari <img src="docs/_static/favicon.ico" width="50" title="cellpose" alt="cellpose" align="right" vspace = "50">

[![Documentation Status](https://readthedocs.org/projects/cellpose-napari/badge/?version=latest)](https://cellpose-napari.readthedocs.io/en/latest/?badge=latest)
[![tests](https://github.com/mouseland/cellpose-napari/workflows/tests/badge.svg)](https://github.com/mouseland/cellpose-napari/actions)
[![codecov](https://codecov.io/gh/Mouseland/cellpose-napari/branch/main/graph/badge.svg)](https://codecov.io/gh/MouseLand/cellpose-napari)
[![PyPI version](https://badge.fury.io/py/cellpose-napari.svg)](https://badge.fury.io/py/cellpose-napari)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/cellpose-napari)](https://pypistats.org/packages/cellpose-napari)
[![Python version](https://img.shields.io/pypi/pyversions/cellpose-napari)](https://pypistats.org/packages/cellpose-napari)
[![License](https://img.shields.io/pypi/l/cellpose-napari.svg?color=green)](https://github.com/mouseland/cellpose-napari/raw/master/LICENSE)
[![Contributors](https://img.shields.io/github/contributors-anon/MouseLand/cellpose-napari)](https://github.com/MouseLand/cellpose-napari/graphs/contributors)
[![website](https://img.shields.io/website?url=https%3A%2F%2Fwww.cellpose.org)](https://www.cellpose.org)
[![GitHub stars](https://img.shields.io/github/stars/MouseLand/cellpose-napari?style=social)](https://github.com/MouseLand/cellpose-napari/)
[![GitHub forks](https://img.shields.io/github/forks/MouseLand/cellpose-napari?style=social)](https://github.com/MouseLand/cellpose-napari/)

a napari plugin for anatomical segmentation of general cellular images

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using with [@napari]'s [cookiecutter-napari-plugin] template.

The plugin code was written by Carsen Stringer, and the cellpose code was written by Carsen Stringer and Marius Pachitariu. To learn about Cellpose, read the [**paper**](https://t.co/kBMXmPp3Yn?amp=1) or watch this [**talk**](https://t.co/JChCsTD0SK?amp=1). 

For support with the plugin, please open an [issue](https://github.com/MouseLand/cellpose-napari/issues). For support with cellpose, please open an [issue](https://github.com/MouseLand/cellpose/issues) on the cellpose repo. 


If you use this plugin please cite the [paper](https://www.nature.com/articles/s41592-020-01018-x):
::
    
      @article{stringer2021cellpose,
      title={Cellpose: a generalist algorithm for cellular segmentation},
      author={Stringer, Carsen and Wang, Tim and Michaelos, Michalis and Pachitariu, Marius},
      journal={Nature Methods},
      volume={18},
      number={1},
      pages={100--106},
      year={2021},
      publisher={Nature Publishing Group}
      }


![cellpose-napari_plugin](docs/_static/napari_main_demo_fast_small.gif?raw=true "cellpose-napari")

## Installation

Install an [Anaconda](https://www.anaconda.com/download/) distribution of Python -- Choose **Python 3** and your operating system. Note you might need to use an anaconda prompt if you did not add anaconda to the path. 

You can install `cellpose-napari` via [pip]:

    pip install cellpose-napari

If install fails in your base environment, create a new environment:
1. Download the [`environment.yml`](https://github.com/MouseLand/cellpose-napari/blob/master/environment.yml?raw=true) file from the repository. You can do this by cloning the repository, or copy-pasting the text from the file into a text document on your local computer.
2. Open an anaconda prompt / command prompt with `conda` for **python 3** in the path
3. Change directories to where the `environment.yml` is and run `conda env create -f environment.yml`
4. To activate this new environment, run `conda activate cellpose-napari`
5. You should see `(cellpose-napari)` on the left side of the terminal line. 

If you have **issues** with cellpose installation, see the [cellpose docs](https://cellpose.readthedocs.io/en/latest/installation.html) for more details, and then if the suggestions fail, open an issue.

### Upgrading software

You can upgrade the plugin with
~~~
pip install cellpose-napari --upgrade
~~~

and you can upgrade cellpose with
~~~
pip install cellpose --upgrade
~~~

### GPU version (CUDA) on Windows or Linux

If you plan on running many images, you may want to install a GPU version of *torch* (if it isn't already installed).

Before installing the GPU version, remove the CPU version:
~~~
pip uninstall torch
~~~

Follow the instructions [here](https://pytorch.org/get-started/locally/) to determine what version to install. The Anaconda install is recommended along with CUDA version 10.2. For instance this command will install the 10.2 version on Linux and Windows (note the `torchvision` and `torchaudio` commands are removed because cellpose doesn't require them):

~~~
conda install pytorch cudatoolkit=10.2 -c pytorch
~~~~

When upgrading GPU Cellpose in the future, you will want to ignore dependencies (to ensure that the pip version of torch does not install):
~~~
pip install --no-deps cellpose --upgrade
~~~

### Installation of github version

Follow steps from above to install the dependencies. In the github repository, run `pip install -e .` and the github version will be installed. If you want to go back to the pip version of cellpose-napari, then say `pip install cellpose-napari`.


## Running the software


Open napari with the cellpose-napari dock widget open
```
napari -w cellpose-napari
```

There is sample data in the File menu, or get started with your own images!

### Detailed usage [documentation](https://cellpose-napari.readthedocs.io/).

## Contributing

Contributions are very welcome. Tests are run with pytest.

## License

Distributed under the terms of the [BSD-3] license,
"cellpose-napari" is free and open source software.

## Dependencies
cellpose-napari relies on the following excellent packages (which are automatically installed with conda/pip if missing):
- [napari](https://napari.org)
- [magicgui](https://napari.org/magicgui/)

cellpose relies on the following excellent packages (which are automatically installed with conda/pip if missing):
- [torch](https://pytorch.org/)
- [numpy](http://www.numpy.org/) (>=1.16.0)
- [numba](http://numba.pydata.org/numba-doc/latest/user/5minguide.html)
- [scipy](https://www.scipy.org/)
- [natsort](https://natsort.readthedocs.io/en/master/)
- [tifffile](https://pypi.org/project/tifffile/)
- [opencv](https://opencv.org/)


[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin
