Settings
--------------------------

There are more settings for cellpose that can be accessed using the CLI or 
through a jupyter notebook. See details at cellpose `docs`_.

Listed are settings available through napari widget. Please submit an 
issue if you would like a new setting available in the widget.

Channels
~~~~~~~~~~~~~~~~~~~~~~~~

Cytoplasm model (`'cyto'`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The cytoplasm model in cellpose is trained on two-channel images, where 
the first channel is the channel to segment, and the second channel is 
an optional nuclear channel. Here are the options for each:
1. 0=grayscale, 1=red, 2=green, 3=blue, 4 ...
2. 0=None (will set to zero), 1=red, 2=green, 3=blue, 4 ...

Nucleus model (`'nuclei'`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The nuclear model in cellpose is trained on two-channel images, where 
the first channel is the channel to segment, and the second channel is 
always set to an array of zeros. Therefore set the first channel as 
0=grayscale, 1=red, 2=green, 3=blue; and set the second channel to 0=None

Diameter 
~~~~~~~~~~~~~~~~~~~~~~~~

The cellpose models have been trained on images which were rescaled 
to all have the same diameter (30 pixels in the case of the `cyto` 
model and 17 pixels in the case of the `nuclei` model). Therefore, 
cellpose needs a user-defined cell diameter (in pixels) as input, or to estimate 
the object size of an image-by-image basis.

The automated estimation of the diameter is a two-step process using the `style` vector 
from the network, a 64-dimensional summary of the input image. We trained a 
linear regression model to predict the size of objects from these style vectors 
on the training data. On a new image the procedure is as follows.

1. Run the image through the cellpose network and obtain the style vector. Predict the size using the linear regression model from the style vector.
2. Resize the image based on the predicted size and run cellpose again, and produce masks. Take the final estimated size as the median diameter of the predicted masks.

Click ``compute diameter from image`` to start diameter estimation.
However, if this estimate is incorrect please set the diameter by hand.

You can also get an estimate from clicking on the image: create a napari 'Shapes' layer 
and draw circles or squares. If you click 
``compute diameter from shape layer``, the plugin will set the diameter to 
the average diameter of the drawn shapes.

Changing the diameter will change the results that the algorithm 
outputs. When the diameter is set smaller than the true size 
then cellpose may over-split cells. Similarly, if the diameter 
is set too big then cellpose may over-merge cells.

Resample
~~~~~~~~~~~~~~~~~~~~~~~~

The cellpose network is run on your rescaled image -- where the rescaling factor is determined 
by the diameter you input (or determined automatically as above). For instance, if you have 
an image with 60 pixel diameter cells, the rescaling factor is 30./60. = 0.5. After determining 
the flows (dX, dY, cellprob), the model runs the dynamics. The dynamics can be run at the rescaled 
size (``resample=False``), or the dynamics can be run on the resampled, interpolated flows 
at the true image size (``resample=True``). ``resample=True`` will create smoother masks when the 
cells are large but will be slower in case; ``resample=False`` will find more masks when the cells 
are small but will be slower in this case. 

Model match threshold
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note there is nothing keeping the neural network from predicting 
horizontal and vertical flows that do not correspond to any real 
shapes at all. In practice, most predicted flows are consistent with 
real shapes, because the network was only trained on image flows 
that are consistent with real shapes, but sometimes when the network 
is uncertain it may output inconsistent flows. To check that the 
recovered shapes after the flow dynamics step are consistent with 
real masks, we recompute the flow gradients for these putative 
predicted masks, and compute the mean squared error between them and
the flows predicted by the network. 

The ``model match threshold`` parameter is inverse of the maximum 
allowed error of the flows for each mask. Decrease this threshold 
if cellpose is not returning as many masks as you'd expect. 
Similarly, increase this threshold if cellpose is returning too many 
ill-shaped masks.

Cell probability threshold
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The network predicts 3 outputs: flows in X, flows in Y, and cell "probability". 
The predictions the network makes of the probability are the inputs to a sigmoid 
centered at zero (1 / (1 + e^-x)), 
so they vary from around -6 to +6. The pixels greater than the 
``cellprob_threshold`` are used to run dynamics and determine masks. The default 
is ``cellprob_threshold=0.0``. Decrease this threshold if cellpose is not returning 
as many masks as you'd expect. Similarly, increase this threshold if cellpose is 
returning too masks particularly from dim areas.

.. _docs: https://cellpose.readthedocs.io/en/latest/command.html#command-line




