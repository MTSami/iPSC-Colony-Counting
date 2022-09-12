import cv2
import os
import matplotlib
import matplotlib.pyplot as plt
import copy
import numpy as np

from PIL import Image,ImageEnhance, ImageFilter 

from scipy import ndimage as ndi
from skimage.feature import peak_local_max
import json


def fix_coordinates(centroids):    
    # Array to store (x, y) swapped as (y, x)
    swapped_coordinates = []
    
    for coord in centroids:
        swapped_coordinates.append((coord[1], coord[0]))
   
    return np.array(swapped_coordinates)



def run_local_maxima(prob_map_img,
                     CONTRAST_FACTOR,
                     LOCAL_MAXIMA_MIN_DIST,
                     LOCAL_MAXIMA_THRESH,
                     RADIUS):
    

    ## Increase contrast of map to help local maxima fingding
    prob_map_pil = Image.fromarray(prob_map_img)
    pil_contraster = ImageEnhance.Contrast(prob_map_pil)
    pil_contrasted_img = pil_contraster.enhance(CONTRAST_FACTOR)
    pil_contrasted_img = pil_contrasted_img.filter(ImageFilter.GaussianBlur(radius = RADIUS))
    prob_map_img = np.array(pil_contrasted_img).astype("uint8")



    # Find local maxima from probability map
    localMax = peak_local_max(prob_map_img, min_distance = LOCAL_MAXIMA_MIN_DIST, threshold_abs = LOCAL_MAXIMA_THRESH)

    # Convert localMax pixels to connect component
    localMax_cc = np.zeros_like(prob_map_img)

    # Fill empty mask
    for coor in localMax:
        cX, cY = coor
        localMax_cc[cX, cY] = 1


    ## Take CC of local maxima map
    output = cv2.connectedComponentsWithStats(localMax_cc)
    (numLabels, labels, stats, centroids) = output

    ## Record image date to json dict
    centroids = centroids [1:, :]
    centroids = fix_coordinates(centroids)
    
    return centroids