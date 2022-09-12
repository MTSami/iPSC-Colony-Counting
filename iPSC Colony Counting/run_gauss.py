import numpy as np
import cv2
import matplotlib.pyplot as plt
import math


def fix_theta(angle, w, h):
    eps = 0.001
    if w < h:
#         print("Width Less")
        inter = 90 - angle
        angle = inter + 180
    else: 
#         print("Width Greater")
        angle = -angle
        temp = w
        w = h
        h = temp
        
    return angle, w+eps, h+eps



def generate_2d_gaussian(x_center = 0, 
                         y_center = 0, 
                         width = 1,
                         height = 1,
                         alpha = 0.2,
                         angle = 0, 
                         out_size = (256, 256)):
    
    """
    x_center: x coordinate of gaussian center
    y_center: y coordinate of gaussian center
    radius: Farthest distance from center of CC to contour
    alpha: parameter to adjust standard deviation of 2D gaussian
    angle: the angle to rotate 2D gaussian axis
    out_size: H x W of output 2D gaussian heatmap
    """
    
    ## Convert to radian
    theta = (2*np.pi*angle) / 360
    
    ## Generate rows and columns labels
    x = np.arange(0, out_size[0], dtype = int)
    y = np.arange(0, out_size[1], dtype = int)

    ## Generate meshgrid
    xv, yv = np.meshgrid(x, y)    
    
    ## Compte sigma for x and y axis    
    x_sigma = width * alpha 
    y_sigma = height * alpha 
    
    
    ## Compute rotation coefficients, according to https://en.wikipedia.org/wiki/Gaussian_function
    
    a = ((np.power(np.cos(theta), 2)) / (2 * np.power(x_sigma, 2))) + \
        ((np.power(np.sin(theta), 2)) / (2 * np.power(y_sigma, 2)))
    
    b = -(np.sin(2 * theta) / (4 * np.power(x_sigma, 2))) + \
         (np.sin(2 * theta) / (4 * np.power(y_sigma, 2)))
    
    c = ((np.power(np.sin(theta), 2)) / (2 * np.power(x_sigma, 2))) + \
        ((np.power(np.cos(theta), 2)) / (2 * np.power(y_sigma, 2)))
    
    ## Generate rotated 2D gaussian map
    gauss_map = np.exp(-( (a*np.power(xv-x_center, 2)) + (2*b*(xv-x_center)*(yv-y_center)) + (c*np.power((yv-y_center),2))))
    
    return gauss_map
    

    
    
def make_gaussian_spread(mask, alpha = 0.3):
    
    """
    mask: binary mask {0, 1}
    alpha: adjustable parameter to change the spread of 2D gaussian
    """
    
    ## List to save all gaussian CCs
    gauss_cc_list = list()
    
    ## Generate CCs
    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    
    ## Loop through each CC
    for i in range(1, numLabels):
        
        single_CC = np.zeros_like(labels)
        single_CC = np.where(labels == i, 1, 0).astype("uint8")
        
        
        ## Generate contour for current CC
        single_CC_contours, _ = cv2.findContours(single_CC, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        
        ## Get rotated, tightest bouding box
        (x, y), (h, w), angle = cv2.minAreaRect(single_CC_contours[0])

        
        ## Adjust bbox angle to gaussian y-axis angle
        angle, w, h = fix_theta(angle, w, h)
        
        ## Generate 2D gaussian map for a single CC
        single_CC_map = generate_2d_gaussian(x_center = int(x), 
                                             y_center = int(y), 
                                             width = w, 
                                             height = h,
                                             alpha = alpha,
                                             angle = angle,
                                             out_size = single_CC.shape)
        ### Apply Mask
        single_CC_map *= single_CC
        single_CC_map[int(y)][int(x)] = 1.0
        
        gauss_cc_list.append(single_CC_map)
        
#         plt.figure(figsize=(10,10))
#         plt.imshow(single_CC_map, cmap = 'jet')
        
        
    ### Handle empty mask
    if len(gauss_cc_list) == 0:
        gt_map = np.zeros_like(image_thresh).astype('float32')
    else:
        gt_map = np.array(gauss_cc_list)
        gt_map = np.max(gt_map, axis = 0)
        
             
    assert np.max(gt_map) <= 1.0, "Probabilty must not exceed 1.0"
    assert np.min(gt_map) >= 0.0, "Probabilty must not be below 0.0"
    assert len(np.where(gt_map == 1)[0]) == numLabels-1, "There should 1.0 at the center of each CC"
    
    return gt_map