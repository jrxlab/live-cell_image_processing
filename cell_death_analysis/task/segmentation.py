import matplotlib.pyplot as pl_
import numpy as np_
import scipy.ndimage as im_
import skimage.morphology as mp_
from task import segmentation_steps as sg_steps_
import os




def CellSegmentation_Original(frame: np_.ndarray,show_sg_steps: bool = False,idx: int=0,
                              n_frames: int=0) -> np_.ndarray:
    """
    CellSegmentation_Original function
    - Applies a wide gaussian filter to remove noise.
    - Watershed x threshold-based binary based segmentation.
    - Applies a binary erosion for a better result.
    It allows to save the segmentation steps in the ouptput directory
    """
    
    # 1.2 = 0.5*hsize(12) / sigma(5)
    
    smooth_frm = im_.gaussian_filter(frame, 5, truncate=1.2) # smoothed image
    
    
    #watershed segmentation
    crest_lines_map = mp_.watershed(-smooth_frm, watershed_line=True) == 0
    
    
    # Threshold-based binary segmentation
    sgm_threshold =3*np_.percentile(np_.fabs(smooth_frm),20) # calculate the threshold 
    segmentation = frame > sgm_threshold
    
    # save the segmentation steps
    if show_sg_steps :
        if idx <= n_frames :
            if not os.path.exists("outputs/segmentation_steps"):
                os.makedirs("outputs/segmentation_steps")
                
            sg_steps_.PlotSegmentationSteps(idx, frame, crest_lines_map, segmentation)
    
    # superpose the watershed result to the thresholding result for a better cell separation
    segmentation[crest_lines_map] = 0  
    
    
    segmentation = mp_.binary_erosion(segmentation, selem=mp_.disk(3)) # binary erosion
    
    
    return segmentation




def NuclSegmentation(frame: np_.ndarray):
    
    """
    NuclSegmentation function
    Segments cell nuclei
    This funcion is empty, it will raise an error if it's used
    """
    
    raise ValueError ("Nuclear segmentation empty !")
    
    
    

def PlotSegmentations(segmentations, show_figure: bool = True) -> None:
    
    """
    PlotSegmentations function
    Plots and saves in the output directory the frame segmentation at each time point 
    """
    
    if not os.path.exists("outputs/segmentation"):
        os.makedirs("outputs/segmentation")
        
        
    if isinstance(segmentations, np_.ndarray):
        pl_.matshow(segmentations)
    
    else:
        def __UpdateFigure__(Frame, figure_, plot_, segmentations_):
            idx = int(round(Frame)) # index
            plot_.set_data(segmentations_[idx]) # set the x and y data
            figure_.canvas.draw_idle() # redraw canvas while idle

        figure = pl_.figure()
        # add axes to the figure
        plot_axes = figure.add_axes([0.1, 0.2, 0.8, 0.65])
        
        
        # plot the values of "segmentations[time_point=0]" as color-coded image.
        for i,seg in enumerate(segmentations):
            plot_axes.matshow(segmentations[i])
            
            # save plot
            pl_.savefig("outputs/segmentation/frame_"+str(i))
        


    if show_figure:
        pl_.show()
     
    