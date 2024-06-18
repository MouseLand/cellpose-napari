import numpy as np
from typing import List

from magicgui import magicgui

from napari import Viewer
from napari.layers import Image
from napari_plugin_engine import napari_hook_implementation

from cellpose_napari.utils import create_table_with_csv_export, MAIN_CHANNEL_CHOICES, OPTIONAL_NUCLEAR_CHANNEL_CHOICES


def widget_wrapper():
  try:
    from torch import no_grad
  except ImportError:
    def no_grad():
      def _deco(func):
          return func
      return _deco

  from napari.qt.threading import thread_worker

  @thread_worker()
  @no_grad()
  def get_cell_counts(images, model_path, channels, flow_threshold=0.4, cellprob_threshold=0):
     from cellpose import models

     CP = models.CellposeModel(pretrained_model=model_path, gpu=True)
     masks, _, _ = CP.eval(
        images,
        channels=channels,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold
     )

     return [np.max(mask) for mask in masks]
  
  @magicgui(
    call_button='get counts',
    layout='vertical',
    model_path = dict(widget_type='FileEdit', label='cellpose model path: ', tooltip='specify model path here'),
    main_channel = dict(widget_type='ComboBox', label='channel to segment', choices=MAIN_CHANNEL_CHOICES, value=0, tooltip='choose channel with cells'),
    optional_nuclear_channel = dict(widget_type='ComboBox', label='optional nuclear channel', choices=OPTIONAL_NUCLEAR_CHANNEL_CHOICES, value=0, tooltip='optional, if available, choose channel with nuclei of cells'),
  )
  def widget(
    viewer: Viewer,
    image_layers: List[Image],
    model_path,
    main_channel,
    optional_nuclear_channel
  ):
      images = []
      for layer in image_layers:
        images.append(layer.data)

      def calculation_finished_callback(cell_counts):
        assert len(image_layers) == len(cell_counts)
        n = len(image_layers)

        table_data = []
        for i in range(n):
           table_data.append([image_layers[i].name, cell_counts[i]])
        
        result_widget = create_table_with_csv_export(["Name", "Cell count"], table_data)
        viewer.window.add_dock_widget(result_widget, name="cell counting result")

      cp_worker = get_cell_counts(images, str(model_path.resolve()), [max(0, main_channel), max(0, optional_nuclear_channel)])
      cp_worker.returned.connect(calculation_finished_callback)
      cp_worker.start()
  
  return widget


@napari_hook_implementation()
def napari_experimental_provide_dock_widget():
    return widget_wrapper, {'name': 'cell counting'}

