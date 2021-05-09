import io
import os
import scipy.misc
import numpy as np
import six
import time
import glob
from IPython.display import display
from six import BytesIO

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


class LiveModel():

    def __init__(self):
        self.default_model_path = f'/home/crimepredictor/modeliai/saved_model'
        self.default_labelmap_path = '/home/crimepredictor/modeliai ir kita/Efdet/automl/efficientdet/data/label_map.pbtxt'
    def load(self,model_path=None, labelmap_path=None):

        if model_path is None:
            model_path = self.default_model_path
        self.model = tf.saved_model.load(model_path)

        if labelmap_path is None:
            labelmap_path = self.default_labelmap_path
        self.category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)

    def run_on_single_frame(self, frame):
        image = np.asarray(frame)
        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis,...]

        # Run inference
        model_fn = self.model.signatures['serving_default']
        output_dict = model_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(output_dict.pop('num_detections'))
        output_dict = {key:value[0, :num_detections].numpy()
                        for key,value in output_dict.items()}
        output_dict['num_detections'] = num_detections

        # detection_classes should be ints.
        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

        # Handle models with masks:
        if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                          output_dict['detection_masks'], output_dict['detection_boxes'],
                           image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                                   tf.uint8)
            output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            self.category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8)

        return frame, output_dict["detection_classes"], output_dict["detection_scores"]

if __name__ == "__main__":
    live_model = LiveModel()
    live_model.load()
    image_path = np.random.choice(glob.glob('/home/crimepredictor/modeliai ir kita/foto/*/*.jpg'))
    img_data = tf.io.gfile.GFile(image_path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    image =  np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)
    im_w_bbox, classes, scores = live_model.run_on_single_frame(image)
    plt.imshow(im_w_bbox)
    plt.savefig("myimage.png")
    print(classes)
    print(scores)
