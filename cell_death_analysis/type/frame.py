from __future__ import annotations

from type.cell import cell_t
from task import segmentation as sg_
from run_parameters import segm_protocol

import numpy as np_
import skimage.feature as ft_
import skimage.measure as ms_
import skimage.morphology as mp_
import skimage.transform as tf_

import matplotlib.pyplot as pl_

class frame_t:
    
    def __init__(self) -> None:
        
        self.channel = ""  # Name of the channel of the frame
        self.time_point = -1  # 0 and up: time point of the frame
        self.size = None  # (height, width) of the frame
        self.contents = None  # numpy.ndarray of the frame contents
        self.cells = None  # List of segmented cells

    @classmethod
    def WithProperties(
        cls, channel: str, time_point: int, contents: np_.ndarray
    ) -> frame_t:
        
        """
        WithProperties method
        Initiates the frame properties
        """
        
        instance = cls()

        instance.channel = channel # channel name
        instance.time_point = time_point # the frame time point
        instance.size = contents.shape # frame size
        instance.contents = contents # frame contents
        
        
        return instance

    def ClearContents(self):
        
        """
        ClearContents method
        To free memory up when all the cell properties are saved
        """
        
        self.contents = None  # To free some memory up 
        

    def RegisterOn(self, ref_frame: np_.ndarray) -> None: 
        
        """
        RegisterOn method (target frame, reference frame)
        - Calculates the between channel shift using the python "register_translation" method
        - Gets the transformation (translation) by the python "EuclideanTransform" method
        - Applies the transformation to the target frame using the python "warp" method
        """
        
        # calculate the shift
        shift = ft_.register_translation( 
            ref_frame,
            self.contents,
            upsample_factor=8,
            space="real",
            return_error=False,
        )
        # get the transformation: translation by the calculated shift
        translation = tf_.EuclideanTransform(translation=shift) 
        
        self.contents = tf_.warp(self.contents, translation) # warped image
        
       
        
                
    def SegmentCells(self, method: str = "cyto",show_sg_steps:bool = False, idx:int=0,
                     n_frames: int=0) -> None:
        
        """
        SegmentCells method
        - Segments cells using the "CellSegmentation_Original" function
          from the "segmentation" script.
        - Gets the cell properties 
        """
        
        if method == "cyto":
            
            segmentation = sg_.CellSegmentation_Original(self.contents,show_sg_steps,idx,n_frames) # call the "CellSegmentation_Original" fct from 
                                                                                                   # "segmentation" script
        
            # Segmentation labeling and cell instantiations
            labeled_sgm = mp_.label(segmentation, connectivity=1) #Label connected regions:
                                                                  #Two pixels are connected when they are neighbors 
                                                                  #and have the same value
            cell_props = ms_.regionprops(labeled_sgm) # measure properties of labeled image regions 
            self.cells = []
            for uid, props in enumerate(cell_props):               
                 # cell properties
                #if props.area >600: 
                coords = (props.coords[:, 0], props.coords[:, 1]) # get pixel coordinates
                cell = cell_t.WithProperties(uid, self.time_point, props.centroid, coords)
                cell.features["area"] = props.area # save the cell area in the features dictionary (see "cell" script)
                
                self.cells.append(cell) # save the cell properties 
             
            
        elif method == "nucl":
            segmentation = sg_.NuclSegmentation(self.contents) # call the " NuclSegmentation" function from "segmentation script
        
        else:
            raise ValueError(f"{method}: Invalid segmentation method")

        

    def CellPositions(self):
        
        """
        CellPositions method
        Calculates the position of the cell centroid
        """
        
        if self.cells is None:
            raise RuntimeError(
                "Segmentation-related function called before segmentation"
            )

        positions = np_.empty((self.cells.__len__(), 2), dtype=np_.float64) # create an empty array

        for idx, cell in enumerate(self.cells):
            positions[idx, :] = cell.position # get the position of the cell centroid in the frame 

        return positions
    

    def CellSegmentation(self, binary: bool = False) -> np_.ndarray:
        
        """
        CellSegmentation function
        Gets the list of the segmented cell composing pixels
        """
        
        if self.cells is None:
            raise RuntimeError(
                "Segmentation-related function called before segmentation"
            )

        segmentation = np_.zeros(self.size, dtype=np_.uint16)

        if binary: # if binary segmentation
            for cell in self.cells: # for each cell in the segmented cells list
                segmentation[cell.pixels] = 1  #segmentation of pixels containing the current cell is = 1 
                                               #(all pixels containing cells in the segmentation array will = 1)
        else:
            for cell in self.cells:
                segmentation[cell.pixels] = cell.uid + 1 # 0 reserved for background
                                                         # for each pixel containing cell, 
                                                         # segmentation = uid of the cell

        return segmentation
