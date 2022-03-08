#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
from sklearn.model_selection import train_test_split
from skimage import io
import cv2
import sklearn
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.color
import skimage.filters
from scipy import ndimage as ndi
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
from skimage.feature import canny
import skimage.io
from skimage import *
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.segmentation import chan_vese
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)


from skimage.segmentation import random_walker
from skimage.data import binary_blobs
from skimage.exposure import rescale_intensity
import skimage


def Thresholding(left, thres = 0.3):
    # blur the image to denoise
    grayImage = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    blurred_image = skimage.filters.gaussian(grayImage, sigma=1.0)
    t = 0.3
    binary_mask = blurred_image < t
    binary_mask = binary_mask.astype('uint8')
    return binary_mask

def Canny_edge(left): 
    grayImage = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    edges = canny(grayImage)
    predict = ndi.binary_fill_holes(edges)
    predict = predict.astype('uint8')
    return predict

def K_means_clustering(left): 
    vectorized = left.reshape((-1,1))
    kmeans = KMeans(n_clusters=2, random_state = 0, n_init=5).fit(vectorized)
    centers = np.uint8(kmeans.cluster_centers_)
    segmented_data = centers[kmeans.labels_.flatten()]
    segmented_image = segmented_data.reshape((left.shape))
    
    grayImage = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)
    predict = blackAndWhiteImage > 0
    predict = predict.astype('uint8')
    
    
    return predict

def chanVese(left):
    
    grayImage = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    predict = chan_vese(grayImage, mu=0.25, lambda1=1, lambda2=1, tol=1e-3,
                dt=0.5, init_level_set="checkerboard",
               extended_output=True)[0]
    predict = predict.astype('uint8')  

    return predict

def waterShed(left): 
    
    img = left
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
    # sure background area
    sure_bg = cv.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv.watershed(img,markers)
    img[markers == -1] = [0,0,0]

    # change image to binary image
    im_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv.threshold(im_gray, 128, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    thresh = 127
    im_bw = cv.threshold(im_gray, thresh, 255, cv.THRESH_BINARY)[1]
    predict = im_bw > 0
    predict = predict.astype('uint8')  
    
    return predict

import numpy as np
import matplotlib.pyplot as plt

from skimage.segmentation import random_walker
from skimage.data import binary_blobs
from skimage.exposure import rescale_intensity
import skimage

def Snakes(left, sigma=0.35, beta= 10):
    # Morphological ACWE
    left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    image = left

    # Initial level set
    init_ls = checkerboard_level_set(image.shape, 6)
    
    ls = morphological_chan_vese(image, init_level_set=init_ls,iterations=35,
                                 smoothing=3)
    inverted = util.invert(ls)
    inverted = inverted > -2
    inverted = inverted.astype('uint8')
    
    return inverted

def walk(data, sigma=0.35, beta= 10):
    rng = np.random.default_rng()
    data = skimage.color.rgb2gray(data)
    sigma = 0.35
    data += rng.normal(loc=0, scale=sigma, size=data.shape)
    data = rescale_intensity(data, in_range=(-sigma, 1 + sigma),
                             out_range=(-1, 1))

    # The range of the binary image spans over (-1, 1).
    # We choose the hottest and the coldest pixels as markers.
    markers = np.zeros(data.shape, dtype=np.uint)
    markers[data < -0.95] = 1
    markers[data > 0.95] = 2

    # Run random walker algorithm
    labels = random_walker(data, markers, beta=10, mode='bf')
    predict = labels>1
    predict = predict.astype('uint8')

    return predict

#weighted sum
def weighted_sum(Region1, Region2, Float_point):
   
    Mask = np.array(Region1) * Float_point + np.array(Region2) * (1-Float_point)
    Mask[Mask <= 0.5] = 0
    Mask[Mask > 0.5] = 1
    
    return Mask

def ADD(region1, region2):
    Mask = region1 + region2
    Mask[Mask > 1] = 1
    
    return Mask


def ADD2(region1, region2):
    Mask = region1 + region2
    Mask[Mask != 2] = 0
    Mask[Mask == 2] = 1
    return Mask

