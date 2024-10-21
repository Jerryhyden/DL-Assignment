# -*- coding: utf-8 -*-
"""detect-objects-with-faster-r-cnn.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Okn9vu4qxUi8qfo8q0r2XOlmk-EQuwQi
"""

!pip install pycocotools

!pip install gluoncv

!pip install torch==1.13.1

"""**IMPORT**"""

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import gluoncv
from gluoncv import model_zoo, data, utils
from pathlib import Path
import random,os
from skimage import io
from pycocotools.coco import COCO
import matplotlib.patches as patches
import time
from tqdm import tqdm
import matplotlib.patches as mpatches

import mxnet as mx

"""**SET UP COCO DATAS**"""

DATA_PATH = "/kaggle/input/coco-2017-dataset/coco2017/val2017/"

# select which set you are going to use (here I used validation set)
annFile = Path('/kaggle/input/coco-2017-dataset/coco2017/annotations/instances_val2017.json')
coco = COCO(annFile)
imgIds = coco.getImgIds() # load all validation set ids

# function to get random image from validation set and it's annotaions through COCO API
def get_rand_img():
    img_id = random.choice(imgIds)
    img_metadata = coco.loadImgs([img_id])
    img = io.imread(DATA_PATH+img_metadata[0]['file_name'])
    annIds = coco.getAnnIds(imgIds=[img_id])
    anns = coco.loadAnns(annIds)
    return img,anns

"""**MODEL**"""

def test_model(model,img):

    # preprocessing
    mx_img = mx.nd.array(img)
    x,original_img = data.transforms.presets.rcnn.transform_test(mx_img)

    # time measurement
    start = time.time()
    box_ids, scores, bboxes = model(x)
    t = time.time() - start

    # image with model detections boxes, scores and ids
    ax = utils.viz.plot_bbox(original_img, bboxes[0], scores[0], box_ids[0], class_names=model.classes)
    return ax,t

"""**Faster R-CNN (Two stage)**"""

rcnn = model_zoo.get_model('faster_rcnn_resnet50_v1b_coco', pretrained=True)
# https://cv.gluon.ai/model_zoo/detection.html

# bounding box is defined by four values in pixels [x_min, y_min, width, height]
def display_ground_truth(image,boxes):
    cpy_img = image.copy()
    fig, ax = plt.subplots()
    ax.imshow(cpy_img)
    for box in boxes:
        rect = patches.Rectangle((int(box['bbox'][0]), int(box['bbox'][1])), int(box['bbox'][2]), int(box['bbox'][3]), linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    ax.set_title("Ground Truth")
    plt.show()

img,anns = get_rand_img()
plt.imshow(img)
display_ground_truth(img,anns)

ax,t = test_model(rcnn,img)
ax.set_title(f"Faster R-CNN \n time taken {round(t,4)}",)
plt.show()

"""**Test  models**"""

img,anns = get_rand_img()
display_ground_truth(img,anns)

ax1,t1 = test_model(rcnn,img)
ax1.set_title(f"Faster R-CNN \n time taken {round(t1,4)}",)
plt.show()

"""**Test on different Dataset**

"""

def generate_img():
    PATH = "/kaggle/input/pascal-voc-2012/VOC2012/JPEGImages/"
    img_path = random.choice(os.listdir(PATH))
    rand_img = io.imread(PATH+img_path)
    return rand_img

img = generate_img()
ax1,t1 = test_model(rcnn,img)
ax1.set_title(f"Faster R-CNN \n time taken {round(t1,4)}",)
plt.show()

img_path = "/kaggle/input/test-model-by-me/OIP.jpg"
img = cv2.imread(img_path)
ax1,t1 = test_model(rcnn,img)
ax1.set_title(f"Faster R-CNN \n time taken {round(t1,4)}",)
plt.show()

img_path = "/kaggle/input/test-model-by-me/test4.jpg"
img = cv2.imread(img_path)
ax1,t1 = test_model(rcnn,img)
ax1.set_title(f"Faster R-CNN \n time taken {round(t1,4)}",)
plt.show()

img_path = "/kaggle/input/test-model-by-me/test5.jpg"
img = cv2.imread(img_path)
ax1,t1 = test_model(rcnn,img)
ax1.set_title(f"Faster R-CNN \n time taken {round(t1,4)}",)
plt.show()

img_path = "/kaggle/input/test-model-by-me/test8.jpg"
img = cv2.imread(img_path)
ax1,t1 = test_model(rcnn,img)
ax1.set_title(f"Faster R-CNN \n time taken {round(t1,4)}",)
plt.show()

img_path = "/kaggle/input/test-model-by-me/test 9.PNG"
img = cv2.imread(img_path)
ax1,t1 = test_model(rcnn,img)
ax1.set_title(f"Faster R-CNN \n time taken {round(t1,4)}",)
plt.show()
