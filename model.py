#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np

"""
Created on Mon Jun 11 15:16:10 2018

@author: huanhuan
"""

import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from collections import defaultdict
from io import StringIO
from PIL import Image
from PIL.Image import fromarray

from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import img_to_array
import keras.backend as K

import pandas as pd
from sklearn.externals import joblib
from skimage import io


print('Loading model files')

## upload the model for inceptionV3 model for similarity study
# using pretrained Inception-V3 model trained on ImageNet
#base_modelv3 = InceptionV3(weights='imagenet',include_top = False)
base_modelv3 = InceptionV3(weights='imagenet')

global keras_model
keras_model = Model(inputs=base_modelv3.input, outputs=base_modelv3.get_layer('avg_pool').output)
global keras_graph
keras_graph = tf.get_default_graph()

# This is needed since the notebook is stored in the object_detection folder.
#sys.path.append("..")
#sys.path.append("./object_detection/")
from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

## model preparation
from object_detection.utils import label_map_util
#from object_detection.utils import visualization_utils as vis_util
MODEL_NAME = 'mask_rcnn_inception_v2_sunglasses_2018_06_12'
PATH_TO_CKPT =  os.path.join('object_detection', MODEL_NAME + '/frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join('object_detection/data', 'sunglasses_label_map.pbtxt')
NUM_CLASSES = 1

## load the model from tensorflow
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
    
## loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
  
def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


## reshape the images
def scale(image, max_size, method=Image.ANTIALIAS):
    """
    resize 'image' to 'max_size' keeping the aspect ratio 
    and place it in center of white 'max_size' image 
    """
    im_aspect = float(image.size[0])/float(image.size[1])
    out_aspect = float(max_size[0])/float(max_size[1])
    if im_aspect >= out_aspect:
        scaled = image.resize((max_size[0], int((float(max_size[0])/im_aspect) + 0.5)), method)
    else:
        scaled = image.resize((int((float(max_size[1])*im_aspect) + 0.5), max_size[1]), method)
 
    offset = (int((max_size[0] - scaled.size[0]) / 2), int((max_size[1] - scaled.size[1]) / 2))
    
    back = Image.new("RGB", max_size, "white")
    back.paste(scaled, offset)
    return back

# preprocessing the images
def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2
    return x

#process input images
def process_input(input_images):
    with keras_graph.as_default():
        images = []
        for image in input_images:
            # , target_size = (299, 299)
            image = scale(image, (299,299))
            #convert the image pixels to a numpy array
            image = img_to_array(image)
            #reshape data for the model
            image = np.expand_dims(image, axis = 0)
            #prepare the image for the model
            image = preprocess_input(image)
            #get image id
            image_id = 'target'
            feature = keras_model.predict(image).ravel()
            images.append((image_id, image, feature))
        return images

##loading the saved model
loaded_model = joblib.load('object_detection/nbrs_joblib.pkl')
sunglasses_db= pd.read_csv('object_detection/sunglasses.csv')
sunglasses_db['ID'] = sunglasses_db["number"].map(str) + '_' + sunglasses_db["style"].map(str) + "_1"
table = pd.read_csv('object_detection/index_table2.csv')


#get the "ID" of all the neigbors
def udfsimular(indices, table):
    neighbours = []
    for i in range(len(indices)):
        t = indices[i]
        idv = table[table.index == t].iloc[0]['ID']
        neighbours.append(idv)
    return neighbours

def udfidfpathh(ids,sunglasses_db, images_db):
    urls = []
    titles = []
    numbers = []
    styles = []
    for i in range(len(ids)): 
        t = ids[i]
        url = sunglasses_db[sunglasses_db.ID == t].iloc[0]['url']
        title = sunglasses_db[sunglasses_db.ID == t].iloc[0]['title']
        number  = sunglasses_db[sunglasses_db.ID == t].iloc[0]['number']
        style  = sunglasses_db[sunglasses_db.ID == t].iloc[0]['style']
        urls.append(url)
        titles.append(title)
        numbers.append(number)
        styles.append(style)
    return urls, titles, numbers, styles

def show5recommendations(name, table, NearestN, input_images, sunglasses_db, images_db):
    input_images_PIL = [Image.fromarray(image) for image in input_images]
    input_image = pd.DataFrame(process_input(input_images_PIL),columns = ['ID', 'image', 'feature'])
    K.clear_session()
    key = list(input_image.loc[:, 'feature'])
    distances, indices = NearestN.kneighbors(key)
    listindices = pd.DataFrame(indices).values.tolist()
    listindices2 = listindices[0]
    ids = udfsimular(listindices2, table)
    urls, titles, numbers, styles = udfidfpathh(ids,sunglasses_db, images_db)
    return urls, titles, numbers, styles


# Given a number image, return the product and style codes
def find_closest(image_np, recommendation_count = 5):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)
    
    # Visualization of the results of a detection.
    instance_masks=output_dict.get('detection_masks')
    instance_scores = output_dict.get('detection_scores')
    instance_boxes = output_dict.get('detection_boxes')
    score_threshold = 0.80
    input_images = []
    for mask, score, box in zip(instance_masks, instance_scores, instance_boxes): 
        if score > score_threshold:
            mask3 = np.stack((mask, mask, mask), -1)
            y_min = int(box[0]*image_np.shape[0])
            y_max = int(box[2]*image_np.shape[0])
            x_min = int(box[1]*image_np.shape[1])
            x_max = int(box[3]*image_np.shape[1])
            segmented_image = mask3*image_np + 255*np.ones_like(image_np)*(1-mask3)
            segmented_image_cropped = segmented_image[y_min:y_max,x_min:x_max]
            input_images = [segmented_image_cropped]
    
    database_directory ='./images'
    if input_images: 
        urls, titles, numbers, styles = show5recommendations('sunglasses', table, loaded_model, input_images, sunglasses_db, database_directory)
        
        #product_ids = [7445279, 9043344, 8590094, 8645251, 7445279]
        #style_ids = [703917, 4314684, 4260093, 4292928, 703917]
        return urls, titles, numbers, styles # product_ids, style_ids
    else: 
        return None, None, None, None 

print('Finished loading model files')




