import numpy as np
import cv2
import napari

class LabelPainter:
    def __init__(self, viewer, labels_layer, points_layer, point_size=10):
        self.viewer = viewer
        self.labels_layer = labels_layer
        self.points_layer = points_layer
        self.point_size = point_size

        self.start_point = None
        self.path = []
        self.drawing = False
        self.moved_outside_start_radius = False

        # Register the callbacks and store their indices
        self.viewer.mouse_drag_callbacks.append(self.handle_mouse_drag)
        self.mouse_drag_callback_index = len(self.viewer.mouse_drag_callbacks) - 1
        
        self.viewer.mouse_move_callbacks.append(self.track_mouse)
        self.mouse_move_callback_index = len(self.viewer.mouse_move_callbacks) - 1

    def clamp_point_to_bounds(self, point, shape):
        """Clamp the point coordinates to be within the bounds of the layer."""
        x, y = point
        max_x, max_y = shape[-2] - 1, shape[-1] - 1
        clamped_x = min(max(x, 0), max_x)
        clamped_y = min(max(y, 0), max_y)
        return clamped_x, clamped_y

    def handle_mouse_drag(self, viewer, event):
        # Ctrl + Left Click to erase labels
        if event.button == 1 and 'Control' in event.modifiers:
            cursor_position = self.labels_layer.world_to_data(event.position)[:2]
            cursor_position = self.clamp_point_to_bounds(cursor_position, self.labels_layer.data.shape)
            cursor_position = np.round(cursor_position).astype(int)

            # Get the label ID under the cursor
            label_id = self.labels_layer.get_value(cursor_position)
            if label_id is not None and label_id != 0:
                # Replace all pixels with this label ID with 0
                self.labels_layer.data[self.labels_layer.data == label_id] = 0
                self.labels_layer.refresh()

        # Right Click to start drawing
        elif event.button == 2 and not self.drawing:
            self.start_point = self.labels_layer.world_to_data(event.position)[:2]
            # Clamp to bounds
            self.start_point = self.clamp_point_to_bounds(self.start_point, self.labels_layer.data.shape)
            self.path = [self.start_point]
            self.points_layer.current_face_color = 'red'
            self.points_layer.current_size = self.point_size * 3
            self.points_layer.add(self.start_point)
            self.drawing = True
            self.moved_outside_start_radius = False
            yield

            while self.drawing:
                yield  # Keep the generator alive until the mouse button is released

            # Reset when the right mouse button is released
            self.start_point = None
            self.path = []
            self.moved_outside_start_radius = False

    def track_mouse(self, viewer, event):
        if self.drawing:
            current_point = self.labels_layer.world_to_data(event.position)[:2]
            current_point = self.clamp_point_to_bounds(current_point, self.labels_layer.data.shape)
            self.path.append(current_point)
            self.points_layer.current_face_color = 'white'
            self.points_layer.current_size = self.point_size
            self.points_layer.add(current_point)

            # Check if the mouse has moved outside the start point's tolerance
            if not self.moved_outside_start_radius and not np.allclose(self.start_point, current_point, atol=self.point_size):
                self.moved_outside_start_radius = True

            # Allow closing the path only if the mouse has moved outside the start radius
            if self.moved_outside_start_radius and np.allclose(self.start_point, current_point, atol=self.point_size):
                self.drawing = False
                self.add_mask(self.path)
                self.points_layer.data = np.empty((0, 2))
                self.points_layer.selected_data = np.empty((0, 1))
                self.start_point = None
                self.path = []

    def add_mask(self, path):
        points = np.array(path)

        # Find the next available label ID
        new_label = self.labels_layer.data.max() + 1

        # Extract the x and y coordinates
        vr = points[:, 1]
        vc = points[:, 0]

        # Create an empty mask
        mask = np.zeros((self.labels_layer.data.shape[-2], self.labels_layer.data.shape[-1]), np.uint8)

        # Get points inside the drawn path
        pts = np.stack((vr, vc), axis=-1)[:, np.newaxis, :]
        mask = cv2.fillPoly(mask, [pts.astype(np.int32)], (255, 0, 0))

        # Add the new label to the label layer only on pixels with value 0
        self.labels_layer.data[(mask > 0) & (self.labels_layer.data == 0)] = new_label
        self.labels_layer.refresh()

    def disconnect(self):
        """Remove the registered callbacks."""
        if self.mouse_drag_callback_index >= 0:
            self.viewer.mouse_drag_callbacks.pop(self.mouse_drag_callback_index)
        if self.mouse_move_callback_index >= 0:
            self.viewer.mouse_move_callbacks.pop(self.mouse_move_callback_index)

# Example usage within a plugin
def activate_label_painter(viewer, image_layer, point_size=4):
    # Determine the size of the labels layer based on the image layer
    labels_layer = viewer.add_labels(np.zeros(image_layer.data.shape[:2], dtype=int), name="Annotations")
    points_layer = viewer.add_points(np.empty((0, 2)), name="Path Points", size=point_size)
    painter = LabelPainter(viewer, labels_layer, points_layer, point_size)
    return painter, labels_layer

def main():
    from skimage import data
    
    # Generate a random image with blobs
    blobs_image = data.binary_blobs(length=512, blob_size_fraction=0.1, n_dim=2).astype(float)

    # Create a Napari viewer
    viewer = napari.Viewer()

    # Add the blobs image to the viewer as an image layer
    image_layer = viewer.add_image(blobs_image, name="Blobs Image")

    # Activate the label painter on this image layer
    return activate_label_painter(viewer, image_layer)

if __name__ == '__main__':
    painter, labels_layer = main()

    # napari.run()
