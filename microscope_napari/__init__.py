try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .widgets.segmentation_widget import napari_experimental_provide_dock_widget
from .samples.cell_data import napari_provide_sample_data