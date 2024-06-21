import os
import sys
from pathlib import Path
from math import isclose
from typing import Callable

import microscope_napari
import napari
import pytest
import torch # for ubuntu tests on CI, see https://github.com/pytorch/pytorch/issues/75912
from microscope_napari.widgets.segmentation_widget import widget_wrapper

# this is your plugin name declared in your napari.plugins entry point
PLUGIN_NAME = "microscope-napari"
# the name of your widget(s)
WIDGET_NAME = "cellpose"

SAMPLE = Path(__file__).parent / "sample.tif"

@pytest.fixture
def viewer_widget(make_napari_viewer: Callable[..., napari.Viewer]):
    viewer = make_napari_viewer()
    _, widget = viewer.window.add_plugin_dock_widget(
        plugin_name=PLUGIN_NAME, widget_name=WIDGET_NAME
    )
    return viewer, widget

def test_basic_function(qtbot, viewer_widget):
    viewer, widget = viewer_widget
    assert len(viewer.window._dock_widgets) == 1

    viewer.open_sample(PLUGIN_NAME, 'rgb_2D')
    viewer.layers[0].data = viewer.layers[0].data[0:128, 0:128]

    #if os.getenv("CI"):
    #    return
        # actually running cellpose like this takes too long and always timesout on CI
        # need to figure out better strategy

    widget()  # run segmentation with all default parameters

    def check_widget():
        assert widget.cellpose_layers

    qtbot.waitUntil(check_widget, timeout=60_000)
    # check that the layers were created properly
    assert len(viewer.layers) == 5
    assert "cp_masks" in viewer.layers[-1].name

    # check that the segmentation was proper, should yield 11 cells
    assert viewer.layers[-1].data.max() == 11

@pytest.mark.skipif(sys.platform.startswith('linux'), reason="ubuntu stalls with two cellpose tests")
def test_compute_diameter(qtbot, viewer_widget):
    viewer, widget = viewer_widget
    viewer.open_sample(PLUGIN_NAME, 'rgb_2D')
    viewer.layers[0].data = viewer.layers[0].data[0:128, 0:128]

    # check the initial value of diameter
    assert widget.diameter.value == "30"
    # run the compute diameter from image function
    # check that the diameter value used for segmentation is correct
    with qtbot.waitSignal(widget.diameter.changed, timeout=60_000) as blocker:
        widget.compute_diameter_button.changed(None)

    assert isclose(float(widget.diameter.value), 24.1, abs_tol=10**-1)

@pytest.mark.skipif(sys.platform.startswith('linux'), reason="ubuntu stalls with >1 cellpose tests")
def test_3D_segmentation(qtbot,  viewer_widget):
    viewer, widget = viewer_widget
    viewer.open_sample(PLUGIN_NAME, 'rgb_3D')

    # set 3D processing
    widget.process_3D.value = True

    widget()  # run segmentation with all default parameters

    def check_widget():
        assert widget.cellpose_layers

    qtbot.waitUntil(check_widget, timeout=120_000)
    # check that the layers were created properly
    assert len(viewer.layers) == 5
    assert "cp_masks" in viewer.layers[-1].name

    # check that the segmentation was proper, should yield 7 cells
    assert viewer.layers[-1].data.max() == 7
