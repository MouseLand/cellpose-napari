import cellpose_napari
from cellpose_napari._dock_widget import widget_wrapper
from pathlib import Path
import os

import pytest

# this is your plugin name declared in your napari.plugins entry point
PLUGIN_NAME = "cellpose-napari"
# the name of your widget(s)
WIDGET_NAME = "cellpose"

SAMPLE = Path(__file__).parent / "sample.tif"


@pytest.fixture
def viewer_widget(make_napari_viewer, napari_plugin_manager):
    napari_plugin_manager.register(cellpose_napari, name=PLUGIN_NAME)
    viewer = make_napari_viewer()
    _, widget = viewer.window.add_plugin_dock_widget(
        plugin_name=PLUGIN_NAME, widget_name=WIDGET_NAME
    )
    return viewer, widget


def test_adding_widget_to_viewer(viewer_widget):
    assert viewer_widget[1].native.parent() is not None


def test_basic_function(qtbot, viewer_widget):
    viewer, widget = viewer_widget
    viewer.open_sample('cellpose-napari', 'rgb_2D.png')

    if os.getenv("CI"):
        return
        # actually running cellpose like this takes too long and always timesout on CI
        # need to figure out better strategy
    widget.compute_diameter_button.changed(None)
    widget()  # run segmentation

    def check_widget():
        assert widget.cellpose_layers

    qtbot.waitUntil(check_widget, timeout=30_000)
    assert len(viewer.layers) == 5
    assert "cp_masks" in viewer.layers[-1].name


# @pytest.mark.parametrize("widget_name", MY_WIDGET_NAMES)
# def test_sample_data_with_viewer(widget_name, make_napari_viewer):
#   viewer = make_napari_viewer()
#   viewer.open_sample('cellpose-napari', 'rgb_3D.tif')
#   assert len(viewer.layers) == 1
