"""
cellpose dock widget module
"""
from typing import Any
from napari_plugin_engine import napari_hook_implementation

import time
import numpy as np
import logging

from napari import Viewer
from napari.layers import Image, Shapes
from magicgui import magicgui
import sys

from microscope_napari.utils import CP_STRINGS, MAIN_CHANNEL_CHOICES, OPTIONAL_NUCLEAR_CHANNEL_CHOICES

# initialize logger
# use -v or --verbose when starting napari to increase verbosity
logger = logging.getLogger(__name__)
if '--verbose' in sys.argv or '-v' in sys.argv:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.WARNING)


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

def widget_wrapper():
    from napari.qt.threading import thread_worker
    try:
        from torch import no_grad
    except ImportError:
        def no_grad():
            def _deco(func):
                return func
            return _deco

    @thread_worker
    @no_grad()
    def run_cellpose(image, model_type, custom_model, channels, channel_axis, diameter,
                    resample, cellprob_threshold, 
                    model_match_threshold, do_3D, stitch_threshold):
        from cellpose import models

        flow_threshold = (31.0 - model_match_threshold) / 10.
        if model_match_threshold==0.0:
            flow_threshold = 0.0
            logger.debug('flow_threshold=0 => no masks thrown out due to model mismatch')
        logger.debug(f'computing masks with cellprob_threshold={cellprob_threshold}, flow_threshold={flow_threshold}')
        if model_type=='custom':
            CP = models.CellposeModel(pretrained_model=custom_model, gpu=True)
        else:
            CP = models.CellposeModel(model_type=model_type, gpu=True)
        masks, flows_orig, _ = CP.eval(image, 
                                    channels=channels, 
                                    channel_axis=channel_axis,
                                    diameter=diameter,
                                    resample=resample,
                                    cellprob_threshold=cellprob_threshold,
                                    flow_threshold=flow_threshold,
                                    do_3D=do_3D,
                                    stitch_threshold=stitch_threshold)
        del CP
        if not do_3D and stitch_threshold==0 and masks.ndim > 2:
            flows = [[flows_orig[0][i], 
                      flows_orig[1][:,i],
                      flows_orig[2][i],
                      flows_orig[3][:,i]] for i in range(masks.shape[0])]
            masks = list(masks)
            flows_orig = flows
        return masks, flows_orig

    @thread_worker
    def compute_diameter(image, channels, model_type):
        from cellpose import models

        CP = models.Cellpose(model_type = model_type, gpu=True)
        diam = CP.sz.eval(image, channels=channels, channel_axis=-1)[0]
        diam = np.around(diam, 2)
        del CP
        return diam

    @thread_worker 
    def compute_masks(masks_orig, flows_orig, cellprob_threshold, model_match_threshold):
        import cv2
        from cellpose.utils import fill_holes_and_remove_small_masks
        from cellpose.dynamics import get_masks
        from cellpose.transforms import resize_image

        #print(flows_orig[3].shape, flows_orig[2].shape, masks_orig.shape)
        flow_threshold = (31.0 - model_match_threshold) / 10.
        if model_match_threshold==0.0:
            flow_threshold = 0.0
            logger.debug('flow_threshold=0 => no masks thrown out due to model mismatch')
        logger.debug(f'computing masks with cellprob_threshold={cellprob_threshold}, flow_threshold={flow_threshold}')
        maski = get_masks(flows_orig[3].copy(), iscell=(flows_orig[2] > cellprob_threshold),
                        flows=flows_orig[1], threshold=flow_threshold*(masks_orig.ndim<3))
        maski = fill_holes_and_remove_small_masks(maski)
        maski = resize_image(maski, masks_orig.shape[-2], masks_orig.shape[-1],
                                        interpolation=cv2.INTER_NEAREST)
        return maski 

    @magicgui(
        call_button='run segmentation',  
        layout='vertical',
        model_type = dict(widget_type='ComboBox', label='model type', choices=['cyto', 'nuclei', 'cyto2', 'custom'], value='cyto', tooltip='there is a <em>cyto</em> model, a new <em>cyto2</em> model from user submissions, and a <em>nuclei</em> model'),
        custom_model = dict(widget_type='FileEdit', label='custom model path: ', tooltip='if model type is custom, specify file path to it here'),
        main_channel = dict(widget_type='ComboBox', label='channel to segment', choices=MAIN_CHANNEL_CHOICES, value=0, tooltip='choose channel with cells'),
        optional_nuclear_channel = dict(widget_type='ComboBox', label='optional nuclear channel', choices=OPTIONAL_NUCLEAR_CHANNEL_CHOICES, value=0, tooltip='optional, if available, choose channel with nuclei of cells'),
        diameter = dict(widget_type='LineEdit', label='diameter', value=30, tooltip='approximate diameter of cells to be segmented'),
        compute_diameter_shape  = dict(widget_type='PushButton', text='compute diameter from shape layer', tooltip='create shape layer with circles and/or squares, select above, and diameter will be estimated from it'),
        compute_diameter_button  = dict(widget_type='PushButton', text='compute diameter from image', tooltip='cellpose model will estimate diameter from image using specified channels'),
        cellprob_threshold = dict(widget_type='FloatSlider', name='cellprob_threshold', value=0.0, min=-8.0, max=8.0, step=0.2, tooltip='cell probability threshold (set lower to get more cells and larger cells)'),
        model_match_threshold = dict(widget_type='FloatSlider', name='model_match_threshold', value=27.0, min=0.0, max=30.0, step=0.2, tooltip='threshold on gradient match to accept a mask (set lower to get more cells)'),
        compute_masks_button  = dict(widget_type='PushButton', text='recompute last masks with new cellprob + model match', enabled=False),
        resample_dynamics = dict(widget_type='CheckBox', text='resample dynamics', value=False, tooltip='if False, mask estimation with dynamics run on resized image with diameter=30; if True, flows are resized to original image size before dynamics and mask estimation (turn on for more smooth masks)'),
        process_3D = dict(widget_type='CheckBox', text='process stack as 3D', value=False, tooltip='use default 3D processing where flows in X, Y, and Z are computed and dynamics run in 3D to create masks'),
        stitch_threshold_3D = dict(widget_type='LineEdit', label='stitch threshold slices', value=0, tooltip='across time or Z, stitch together masks with IoU threshold of "stitch threshold" to create 3D segmentation'),
        clear_previous_segmentations = dict(widget_type='CheckBox', text='clear previous results', value=True),
        output_flows = dict(widget_type='CheckBox', text='output flows and cellprob', value=True),
        output_outlines = dict(widget_type='CheckBox', text='output outlines', value=True),
    )
    def widget(#label_logo, 
        viewer: Viewer,
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
        resample_dynamics,
        process_3D,
        stitch_threshold_3D,
        clear_previous_segmentations,
        output_flows,
        output_outlines
    ) -> None:
        # Import when users activate plugin

        if not hasattr(widget, 'cellpose_layers'):
            widget.cellpose_layers = []
        
        if clear_previous_segmentations:
            layer_names = [layer.name for layer in viewer.layers]
            for layer_name in layer_names:
                if any([cp_string in layer_name for cp_string in CP_STRINGS]):
                    viewer.layers.remove(viewer.layers[layer_name])
            widget.cellpose_layers = []

        def _new_layers(masks, flows_orig):
            from cellpose.utils import masks_to_outlines
            from cellpose.transforms import resize_image
            import cv2

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

            # get physical scale (...ZYX)
            if len(image_layer.scale) > 3:
                physical_scale = image_layer.scale[-3:]
            else:
                physical_scale = image_layer.scale
            
            if widget.output_flows.value:
                layers.append(viewer.add_image(flows, name=image_layer.name + '_cp_flows' + widget.iseg, visible=False, rgb=True, scale=physical_scale))
                layers.append(viewer.add_image(cellprob, name=image_layer.name + '_cp_cellprob' + widget.iseg, visible=False, scale=physical_scale))
            if widget.output_outlines.value:
                layers.append(viewer.add_labels(outlines, name=image_layer.name + '_cp_outlines' + widget.iseg, visible=False, scale=physical_scale))
            layers.append(viewer.add_labels(masks, name=image_layer.name + '_cp_masks' + widget.iseg, visible=False, scale=physical_scale))
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
                if not float(stitch_threshold_3D):
                    widget.compute_masks_button.enabled = True            
            except Exception as e:
                logger.error(e)
            widget.call_button.enabled = True
            
        image = image_layer.data 
        # put channels last
        widget.n_channels = 0
        widget.channel_axis = None
        if image_layer.ndim == 4 and not image_layer.rgb:
            chan = np.nonzero([a=='c' for a in viewer.dims.axis_labels])[0]
            if len(chan) > 0:
                chan = chan[0]
                widget.channel_axis = chan
                widget.n_channels = image.shape[chan]
        elif image_layer.ndim==3 and not image_layer.rgb:
            image = image[:,:,:,np.newaxis]
        elif image_layer.rgb:
            widget.channel_axis = -1

        cp_worker = run_cellpose(image=image,
                                model_type=model_type,
                                custom_model=str(custom_model.resolve()),
                                channels=[max(0, main_channel), 
                                            max(0, optional_nuclear_channel)],
                                channel_axis=widget.channel_axis, 
                                diameter=float(diameter),
                                resample=resample_dynamics,
                                cellprob_threshold=cellprob_threshold,
                                model_match_threshold=model_match_threshold,
                                do_3D=(process_3D and float(stitch_threshold_3D)==0 and image_layer.ndim>2),
                                stitch_threshold=float(stitch_threshold_3D) if image_layer.ndim>2 else 0.0)
        cp_worker.returned.connect(_new_segmentation)
        cp_worker.start()


    def update_masks(masks):     
        from cellpose.utils import masks_to_outlines

        outlines = masks_to_outlines(masks) * masks
        if masks.ndim==3 and widget.n_channels > 0:
            masks = np.repeat(np.expand_dims(masks, axis=widget.channel_axis), 
                            widget.n_channels, axis=widget.channel_axis)
            outlines = np.repeat(np.expand_dims(outlines, axis=widget.channel_axis), 
                                widget.n_channels, axis=widget.channel_axis)
        
        widget.viewer.value.layers[widget.image_layer.value.name + '_cp_masks' + widget.iseg].data = masks
        outline_str = widget.image_layer.value.name + '_cp_outlines' + widget.iseg
        if outline_str in widget.viewer.value.layers:
            widget.viewer.value.layers[outline_str].data = outlines
        widget.masks_orig = masks
        logger.debug('masks updated')


    @widget.compute_masks_button.changed.connect 
    def _compute_masks(e: Any):
        
        mask_worker = compute_masks(widget.masks_orig, 
                                    widget.flows_orig, 
                                    widget.cellprob_threshold.value, 
                                    widget.model_match_threshold.value)
        mask_worker.returned.connect(update_masks)
        mask_worker.start()

    def _report_diameter(diam):
        widget.diameter.value = diam
        logger.debug(f'computed diameter = {diam}')
    
    @widget.compute_diameter_button.changed.connect 
    def _compute_diameter(e: Any):

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
    def _compute_diameter_shape(e: Any):
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
            logger.error('no square or circle shapes created')

    return widget            


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return widget_wrapper, {'name': 'cellpose segmentation'}

