#!/usr/bin/python3
#-*- coding: utf-8 -*-

import matplotlib.pyplot as pl_
import numpy as np_
from scipy.signal import convolve2d
from skimage import exposure, filters
from skimage import img_as_float
import skimage.morphology as mp_


def plot_img_and_hist(image, axes, bins=150):
    """Plot an image along with its histogram and cumulative histogram.

    """
    image = img_as_float(image)
    ax_hist = axes
    #ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image)
    ax_img.set_axis_off()

    # Display histogram
    ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
    ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')
    ax_hist.set_xlim(0, np_.amax(image))
    ax_hist.set_yticks([])

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    ax_cdf.set_yticks([])

    return ax_img, ax_hist, ax_cdf







def ContrastNormalized(frame, percentile : list) -> np_.ndarray:
    
    """
    ContrastNormalized function 
    This function normalizes the background using the contrast streching method,
    the histogram of the pixel intesities is normalized between [min_percentile, max_percentile]
    """
    
   
       
    kernel = np_.array([[1,2,1],[2,4,2],[1,2,1]]) 
    kernel = kernel/np_.sum(kernel) # normalized gaussian filter
    
    #convolution
    edges = convolve2d(frame, kernel, mode='same') 
    

    # contrast stretching
    p_inf = np_.percentile(edges, percentile[0])
    p_sup = np_.percentile(edges, percentile[1])
    img = exposure.rescale_intensity(frame, in_range=(p_inf, p_sup)) #stretching image intensity
    
    
    smooth_frm = filters.gaussian(img, sigma=(3,3), multichannel=None) # smooth image with a gaussian filter [5,3]
    
    
    
   
    
    return smooth_frm



