"""
cellpose dock widget module
"""
from napari_plugin_engine import napari_hook_implementation

import sys, pathlib, os, time
import numpy as np
import cv2
from PyQt5.QtWidgets import QWidget, QTextEdit, QVBoxLayout, QLabel, QTextBrowser
from PyQt5 import QtCore, QtGui, QtWidgets

import napari 
import napari.utils.notifications
from napari import Viewer
from napari.layers import Image, Shapes
from napari_plugin_engine import napari_hook_implementation
from magicgui import magicgui, magic_factory

from cellpose.models import CellposeModel, Cellpose
from cellpose.utils import masks_to_outlines, fill_holes_and_remove_small_masks
from cellpose.dynamics import get_masks
from cellpose.transforms import resize_image

from cellpose.__main__ import logger_setup
import logging

logger, log_file = logger_setup()

class TextWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.resize(800,400)
        self.label = QtWidgets.QLabel('keep open to see cellpose run info')
        self.logTextBox = QtWidgets.QPlainTextEdit(self)
        self.logTextBox.setReadOnly(True)
        self.cursor = self.logTextBox.textCursor()
        self.cursor.movePosition(self.cursor.End)    
        
        layout = QtWidgets.QVBoxLayout()
        # Add the new logging box widget to the layout
        layout.addWidget(self.label)
        layout.addWidget(self.logTextBox)
        self.setLayout(layout)
        self.show()

#@thread_worker
def read_logging(log_file, logwindow):
    with open(log_file, 'r') as thefile:
        #thefile.seek(0,2) # Go to the end of the file
        while True:
            line = thefile.readline()
            if not line:
                time.sleep(0.01) # Sleep briefly
                continue
            else:
                logwindow.cursor.movePosition(logwindow.cursor.End)
                logwindow.cursor.insertText(line)
                yield line
            
main_channel_choices = [('average all channels', 0), ('0=red', 1), ('1=green', 2), ('2=blue', 3),
                        ('3', 4), ('4', 5), ('5', 6), ('6', 7), ('7', 8), ('8', 9)]
optional_nuclear_channel_choices = [('none', 0), ('0=red', 1), ('1=green', 2), ('2=blue', 3),
                                    ('3', 4), ('4', 5), ('5', 6), ('6', 7), ('7', 8), ('8', 9)]
cp_strings = ['_cp_masks_', '_cp_outlines_', '_cp_flows_', '_cp_cellprob_']

#logo = os.path.join(__file__, 'logo/logo_small.png')

def widget_wrapper():
    from napari.qt.threading import thread_worker
    
    @thread_worker
    def run_cellpose(image, model_type, custom_model, channels, diameter,
                    net_avg, resample, cellprob_threshold, 
                    model_match_threshold, do_3D, stitch_threshold):
        flow_threshold = (31.0 - model_match_threshold) / 10.
        if model_match_threshold==0.0:
            flow_threshold = 0.0
            logger.info('flow_threshold=0 => no masks thrown out due to model mismatch')
        logger.info(f'computing masks with cellprob_threshold={cellprob_threshold}, flow_threshold={flow_threshold}')
        pretrained_model = custom_model if model_type=='custom' else model_type
        CP = CellposeModel(pretrained_model=pretrained_model, gpu=True)
        masks, flows_orig, _ = CP.eval(image, 
                                    channels=channels, 
                                    channels_last=True,
                                    diameter=diameter,
                                    net_avg=net_avg,
                                    resample=resample,
                                    cellprob_threshold=cellprob_threshold,
                                    flow_threshold=flow_threshold,
                                    do_3D=do_3D,
                                    stitch_threshold=stitch_threshold)
        del CP 
        segmentation = (masks, flows_orig)
        return segmentation

    @thread_worker
    def compute_diameter(image, channels, model_type):
        CP = Cellpose(model_type = model_type, gpu=True)
        diam = CP.sz.eval(image, channels=channels, channels_last=True)[0]
        diam = np.around(diam, 2)
        del CP
        return diam

    @thread_worker 
    def compute_masks(masks_orig, flows_orig, cellprob_threshold, model_match_threshold):
        flow_threshold = (31.0 - model_match_threshold) / 10.
        if model_match_threshold==0.0:
            flow_threshold = 0.0
            logger.info('flow_threshold=0 => no masks thrown out due to model mismatch')
        logger.info(f'computing masks with cellprob_threshold={cellprob_threshold}, flow_threshold={flow_threshold}')
        maski = get_masks(flows_orig[3].copy(), iscell=(flows_orig[2] > cellprob_threshold),
                        flows=flows_orig[1], threshold=flow_threshold)
        if flows_orig[1].ndim < 4:
            maski = fill_holes_and_remove_small_masks(maski)
        maski = resize_image(maski, masks_orig.shape[-2], masks_orig.shape[-1],
                                        interpolation=cv2.INTER_NEAREST)
        return maski 

    @magicgui(call_button='run segmentation',  
            layout='vertical',
            model_type = dict(widget_type='ComboBox', label='model type', choices=['cyto', 'nuclei', 'custom'], value='cyto', tooltip='model type'),
            custom_model = dict(widget_type='FileEdit', label='custom model path: '),
            main_channel = dict(widget_type='ComboBox', label='channel to segment', choices=main_channel_choices, value=0, tooltip='model type'),
            optional_nuclear_channel = dict(widget_type='ComboBox', label='optional nuclear channel', choices=optional_nuclear_channel_choices, value=0, tooltip='model type'),
            diameter = dict(widget_type='LineEdit', label='diameter', value=30),
            compute_diameter_shape  = dict(widget_type='PushButton', text='compute diameter from shape layer'),
            compute_diameter_button  = dict(widget_type='PushButton', text='compute diameter from image'),
            cellprob_threshold = dict(widget_type='FloatSlider', name='cellprob_threshold', value=0.0, min=-8.0, max=8.0, step=0.2),
            model_match_threshold = dict(widget_type='FloatSlider', name='model_match_threshold', value=27.0, min=0.0, max=30.0, step=0.2, tooltip='threshold on gradient match to accept a mask (set lower to get more cells)'),
            compute_masks_button  = dict(widget_type='PushButton', text='recompute last masks with new cellprob + model match', enabled=False),
            net_average = dict(widget_type='CheckBox', text='average 4 nets', value=True),
            resample_dynamics = dict(widget_type='CheckBox', text='resample dynamics', value=False),
            process_3D = dict(widget_type='CheckBox', text='process stack as 3D', value=False, tooltip='use default 3D processing where flows in X, Y, and Z are computed and dynamics run in 3D to create masks'),
            stitch_threshold_3D = dict(widget_type='LineEdit', label='stitch threshold slices', value=0, tooltip='across time or Z, stitch together masks with IoU threshold of "stitch threshold" to create 3D segmentation'),
            clear_previous_segmentations = dict(widget_type='CheckBox', text='clear previous results', value=True),
            output_flows = dict(widget_type='CheckBox', text='output flows and cellprob', value=True),
            output_outlines = dict(widget_type='CheckBox', text='output outlines', value=True),
            )

    def widget(#label_logo, 
            viewer: napari.viewer.Viewer,
                image_layer: Image,
                model_type,
                custom_model,
                main_channel,
                optional_nuclear_channel,
                diameter,
                shape_layer: Shapes,
                compute_diameter_shape,
                compute_diameter_button,
                cellprob_threshold,
                model_match_threshold,
                compute_masks_button,
                net_average,
                resample_dynamics,
                process_3D,
                stitch_threshold_3D,
                clear_previous_segmentations,
                output_flows,
                output_outlines):
        if not hasattr(widget, 'cellpose_layers'):
            widget.cellpose_layers = []
        
        if clear_previous_segmentations:
            layer_names = [layer.name for layer in viewer.layers]
            for layer_name in layer_names:
                logger.info(layer_name)
                if any([cp_string in layer_name for cp_string in cp_strings]):
                    viewer.layers.remove(viewer.layers[layer_name])
            widget.cellpose_layers = []

        def _new_layers(masks, flows_orig):
            flows = resize_image(flows_orig[0], masks.shape[-2], masks.shape[-1],
                                        interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            cellprob = resize_image(flows_orig[2], masks.shape[-2], masks.shape[-1],
                                    no_channels=True)
            cellprob = cellprob.squeeze()
            outlines = masks_to_outlines(masks) * masks  
            if masks.ndim==3 and widget.n_channels > 0:
                masks = np.repeat(np.expand_dims(masks, axis=widget.channel_axis), 
                                widget.n_channels, axis=widget.channel_axis)
                outlines = np.repeat(np.expand_dims(outlines, axis=widget.channel_axis), 
                                    widget.n_channels, axis=widget.channel_axis)
                flows = np.repeat(np.expand_dims(flows, axis=widget.channel_axis), 
                                widget.n_channels, axis=widget.channel_axis)
                cellprob = np.repeat(np.expand_dims(cellprob, axis=widget.channel_axis), 
                                    widget.n_channels, axis=widget.channel_axis)
                
            widget.flows_orig = flows_orig
            widget.masks_orig = masks
            widget.iseg = '_' + '%03d'%len(widget.cellpose_layers)
            layers = []
            if widget.output_flows.value:
                layers.append(viewer.add_image(flows, name=image_layer.name + '_cp_flows' + widget.iseg, visible=False, rgb=True))
                layers.append(viewer.add_image(cellprob, name=image_layer.name + '_cp_cellprob' + widget.iseg, visible=False))
            if widget.output_outlines.value:
                layers.append(viewer.add_labels(outlines, name=image_layer.name + '_cp_outlines' + widget.iseg, visible=False))
            layers.append(viewer.add_labels(masks, name=image_layer.name + '_cp_masks' + widget.iseg, visible=False))
            widget.cellpose_layers.append(layers)

        def _new_segmentation(segmentation):
            masks, flows_orig = segmentation
            try:
                if image_layer.ndim > 2 and not process_3D and not float(stitch_threshold_3D):
                    for mask, flow_orig in zip(masks, flows_orig):
                        _new_layers(mask, flow_orig)
                else:
                    _new_layers(masks, flows_orig)
                
                for layer in viewer.layers:
                    layer.visible = False
                viewer.layers[-1].visible = True
                image_layer.visible = True
                if not float(stitch_threshold_3D) and not image_layer.ndim > 2:
                    widget.compute_masks_button.enabled = True            
            except Exception as e:
                print(e)
            widget.call_button.enabled = True
            
        image = image_layer.data 
        # put channels last
        widget.n_channels = 0
        if image_layer.ndim == 4 and not image_layer.rgb:
            chan = np.nonzero([a=='c' for a in viewer.dims.axis_labels])[0]
            if len(chan) > 0 and chan[0] < 3:
                chan = chan[0]
                widget.channel_axis = chan
                axes = np.arange(0, 4)
                axes[chan:-1] = axes[chan+1:]
                axes[-1] = chan
                image = image.transpose(axes)
            else:
                widget.channel_axis = 3
            widget.n_channels = image.shape[-1]
        elif image_layer.ndim==3 and not image_layer.rgb:
            image = image[:,:,:,np.newaxis]

        print(image.shape)
        cp_worker = run_cellpose(image=image,
                                model_type=model_type,
                                custom_model=str(custom_model.resolve()),
                                channels=[max(0, main_channel), 
                                            max(0, optional_nuclear_channel)], 
                                diameter=float(diameter),
                                net_avg=net_average,
                                resample=resample_dynamics,
                                cellprob_threshold=cellprob_threshold,
                                model_match_threshold=model_match_threshold,
                                do_3D=(process_3D and float(stitch_threshold_3D)==0 and image_layer.ndim>2),
                                stitch_threshold=float(stitch_threshold_3D) if image_layer.ndim>2 else 0.0)
        cp_worker.returned.connect(_new_segmentation)
        cp_worker.start()
        
        
        pass

    def update_masks(maski):
        outlines = masks_to_outlines(maski) * maski  
        widget.viewer.value.layers[widget.image_layer.value.name + '_cp_masks' + widget.iseg].data = maski
        widget.viewer.value.layers[widget.image_layer.value.name + '_cp_outlines' + widget.iseg].data = outlines
        widget.masks_orig = maski
        logger.info('masks updated')


    @widget.compute_masks_button.changed.connect 
    def _compute_masks(event):
        mask_worker = compute_masks(widget.masks_orig, 
                                    widget.flows_orig, 
                                    widget.cellprob_threshold.value, 
                                    widget.model_match_threshold.value)
        mask_worker.returned.connect(update_masks)
        mask_worker.start()

    def _report_diameter(diam):
        widget.diameter.value = diam
        logger.info(f'computed diameter = {diam}')
    
    @widget.compute_diameter_button.changed.connect 
    def _compute_diameter(event):
        print('button clicked')
        if widget.model_type.value == 'custom':
            logger.error('cannot compute diameter for custom model')
        else:
            model_type = widget.model_type.value
            channels = [max(0, widget.main_channel.value), max(0, widget.optional_nuclear_channel.value)],
            image = widget.image_layer.value.data
            diam_worker = compute_diameter(image, channels, model_type)
            diam_worker.returned.connect(_report_diameter)
            diam_worker.start()
        ### TODO 3D images

    @widget.compute_diameter_shape.changed.connect 
    def _compute_diameter_shape(event):
        diam = 0
        k=0
        for d in widget.shape_layer.value.data:
            if len(d)==4:
                diam += np.ptp(d, axis=0)[-2:].sum()
                k+=2
        diam /= k
        diam *= (27/30)
        if k>0:
            _report_diameter(diam)
        else:
            logging.error('no square or circle shapes created')

    return widget            


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return widget_wrapper, {'name': 'cellpose'}


