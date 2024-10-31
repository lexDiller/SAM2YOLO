import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
from pycocotools import mask as maskUtils
import torch
import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


# from pycocotools import mask as maskUtils

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def rle2polygon(segmentation):
    m = maskUtils.decode(segmentation)
    m[m > 0] = 255
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    polygons = []
    for contour in contours:
        epsilon = 0.001 * cv2.arcLength(contour, True)
        contour_approx = cv2.approxPolyDP(contour, epsilon, True)
        polygon = contour_approx.flatten().tolist()
        polygons.append(polygon)
    return polygons


def lala(a):
    x = a[::2]  # координаты x (четные индексы)
    y = a[1::2]  # координаты y (нечетные индексы)
    plt.plot(x, y)
    # plt.figure(figsize=(20, 20))
    plt.title('')
    plt.xlabel('')
    plt.ylabel('')
    plt.axis('off')
    ax = plt.gca()
    ax.set_autoscale_on(None)


sys.path.append("..")
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)



image = cv2.imread('snow10.jpg')
image_white = cv2.imread('white_sheet.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.88,
    stability_score_thresh=0.88,
    crop_n_layers=3,
    crop_n_points_downscale_factor=2,
    min_mask_region_area=500,  # Requires open-cv to run post-processing
    output_mode='coco_rle'
)