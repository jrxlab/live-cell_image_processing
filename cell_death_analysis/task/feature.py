import numpy as np_
from typing import Tuple


def MOMPLoc(
    channel: np_.ndarray, momp: np_.ndarray, roi_coords: Tuple[np_.ndarray, np_.ndarray]
) -> float:
    """
    MOMPLoc function
    Find the MOMP location, by searching for the nucleus coordinates 
    """
    
    local_channel = channel[roi_coords] # Regieon of interest location in the choosed channel
    max_intensity_value = np_.max(local_channel) # get the max intensity value
    max_map_at_coords = (channel == max_intensity_value)[roi_coords] # get the coordinates of the max intensity value
    first_max_idx = np_.nonzero(max_map_at_coords)[0][0] # get the first index of the max intensity value.
    
    row = roi_coords[0][first_max_idx] # the 1st max intensity value row coordinate
    col = roi_coords[1][first_max_idx] # the 1st max intensity value column coordinate
    
    # calculate the MOMP coordinate
    roi_momp = momp[
        max(row - 2, 0) : min(row + 2, momp.shape[0] - 1),
        max(col - 2, 0) : min(col + 2, momp.shape[1] - 1),
    ]

    return np_.percentile(roi_momp, 5)



def Edginess(channel, roi_coords, origin) -> np_.ndarray:
    
    """
    Edginess function
    Gets the cell edges by searching for the max cardinal jump
    """
    
    # determine the mask 
    row_shifts = (-1, 0, 1, -1, 1, -1, 0, 1)
    col_shifts = (-1, -1, -1, 0, 0, 1, 1, 1)
    profile_length = 5

    n_shifts = row_shifts.__len__() # number of shifts
    cardinal_jumps = np_.zeros(n_shifts, dtype=np_.float64)

    #get origins
    origin_row = int(round(origin[0]))
    origin_col = int(round(origin[1]))

    roi_map = np_.zeros(channel.shape, dtype=np_.bool)
    roi_map[roi_coords] = True
    rolling_profile = np_.empty(profile_length, dtype=channel.dtype)
    out_dist_threshold = (profile_length // 2) + 1 # // get the division quotient 
    row_col=[]

    for line_idx in range(n_shifts):
        
        row = origin_row
        col = origin_col
        out_dist = 0
        prof_idx = 0
        rolling_profile.fill(0)

        while out_dist < out_dist_threshold: # while the distance is lower the distance threshold
            prev_intensity = channel[row, col] # previous intensity value

            row += row_shifts[line_idx] # move to the next row
            if (row < 0) or (row >= channel.shape[0]): # if the row value is negative or bigger then the channel row shape 
                break
            col += col_shifts[line_idx] # move to the next col
            if (col < 0) or (col >= channel.shape[1]): # if the col value is negative or bigger then the channel col shape 
                break

            prof_idx += 1 # move to the next profile index
            # save the intensity difference between the previous and the current intensities
            rolling_profile[prof_idx % profile_length] = (
                prev_intensity - channel[row, col]
            )
            row_col.append((row,col))
            
            if not roi_map[row, col]:
                out_dist += 1
        
       
        cardinal_jumps[line_idx] = max(rolling_profile) # get the max cardinal jump
        
    return cardinal_jumps
