import os
from pathlib import Path
from math import isclose
from typing import Callable

import napari
import pytest
import torch # for ubuntu tests on CI, see https://github.com/pytorch/pytorch/issues/75912

# this is your plugin name declared in your napari.plugins entry point
PLUGIN_NAME = "cellpose-napari"
# the name of your widget(s)
WIDGET_NAME = "cellpose"

SAMPLE = Path(__file__).parent / "sample.tif"

@pytest.fixture(autouse=True)
def patch_mps_on_CI(monkeypatch):
    # https://github.com/actions/runner-images/issues/9918
    if os.getenv('CI'):
        monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
        monkeypatch.setattr("cellpose.core.assign_device", lambda **kwargs: (torch.device("cpu"), False))


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

    widget.model_type.value = "cyto3"
    widget()  # run segmentation with all default parameters

    def check_widget():
        assert widget.cellpose_layers

    qtbot.waitUntil(check_widget, timeout=60_000)
    # check that the layers were created properly
    assert len(viewer.layers) == 5
    assert "cp_masks" in viewer.layers[-1].name

    # check that the segmentation was proper, cyto3 yields 10 cells
    assert viewer.layers[-1].data.max() == 10

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

    # local on macOS with MPS, get 20.37, with CPU-only it's 20.83, same as CI
    # so choosing a target that works for both
    assert isclose(float(widget.diameter.value), 20.6, abs_tol=0.3)

def test_3D_segmentation(qtbot,  viewer_widget):
    viewer, widget = viewer_widget
    # by default the widget loads with `process_3D` set to False
    assert widget.process_3D.value == False
    viewer.open_sample(PLUGIN_NAME, 'rgb_3D')

    # check that 3D processing is set correctly after opening a 3D image
    assert widget.process_3D.value == True

    widget.model_type.value = "cyto3"
    widget()  # run segmentation with all default parameters

    def check_widget():
        assert widget.cellpose_layers

    qtbot.waitUntil(check_widget, timeout=120_000)
    # check that the layers were created properly
    assert len(viewer.layers) == 5
    assert "cp_masks" in viewer.layers[-1].name

    # check that the segmentation was proper, `cyto3` should yield 9 cells
    assert viewer.layers[-1].data.max() == 9
