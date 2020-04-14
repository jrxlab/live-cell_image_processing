import matplotlib.pyplot as pl_
import matplotlib.image as img_
from matplotlib.widgets import Slider as mp_slider_t
import numpy as np_
import scipy.ndimage as im_
import skimage.morphology as mp_
import skimage.segmentation as sg



def CellSegmentation_SLIC(frame: np_.ndarray) ->np_.ndarray:

     """"""""
     "superpixelisation method for cell segmentation"
     "segments image using k-means clustering"

     """"""""
     frame+=frame.min()
     frame/=frame.max()
     smooth_frm = im_.gaussian_filter(frame, 3)
    
     segments = sg.slic(smooth_frm*255, n_segments = 850, sigma = 0,compactness=10, enforce_connectivity=True)

     # image with local means in SLIC regions
     segm_empty = np_.empty_like(segments, dtype=np_.float64)

     for label in range(segments.max()+1):
         region = segments==label
         segm_empty[region]=np_.mean(smooth_frm[region])

     # for final segmentation
     segmentation = np_.zeros_like(segments)

     for label in range(segments.max() + 1):
         region = segments == label
         dilated_region = mp_.binary_dilation(region)
         ring_region = np_.logical_and(dilated_region, 
                                       np_.logical_not(region))
         # padding to avoid multiple contacts of region with border(would make the background unconnected)
         labeled_background = mp_.label(np_.logical_not(np_.pad(region,1, mode='constant')), connectivity=1)
         # selection of regions that are local max (with the first condition) 
         #and exclusion of regions which have holes (with the second condition of if loop)
         if (np_.min(segm_empty[region]) > np_.max(segm_empty[ring_region])) and (labeled_background.max() == 1):
             segmentation[region] = label

     segmentation = sg.relabel_sequential(segmentation)[0]
     segmentation = mp_.binary_dilation(segmentation, selem=mp_.disk(7))
     return segmentation
 
    
    
