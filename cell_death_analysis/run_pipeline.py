from task import normalization as nm_
from type.sequence import sequence_t
from analysis.death_time import CellDeath_t
from analysis.process_trajectories import process_traj
import matplotlib.pyplot as pl_
from run_parameters import *
import os

# create an output directory

if not os.path.exists("outputs"):
    os.makedirs("outputs")
            
############# sequence redaing #############

print("Reading Sequence")

sequence = sequence_t.FromTiffFile(
    sequence_path,
    acq_seq,
    from_frame=from_frame,
    to_frame=to_frame, 
    n_channels=n_channels)



############# channel alignement #############

if n_channels >1:
    print("Registration")
    sequence.RegisterChannelsOn(ref_channel.upper()) 


########### contrast normalization #############
    
print("Background Normalization")

post_processing_fct = lambda img: nm_.ContrastNormalized(img,percentile) # to run the contrast normalization script

sequence.BackgroundNormalization(acq_seq,post_processing_fct=post_processing_fct)


#############   SEGMENTATION   #############

print("Segmentation")

sequence.SegmentCellsOnChannel(segm_channel.upper(),show_sg_steps,n_frames)

print(sequence)

sequence.PlotSegmentations() # plot the segmentation result


#############   TRACKING   #################

print("Tracking Feature Computation")

sequence.ComputeCellFeatures(acq_seq,channel_treat_prot.lower(),fret_ratio,edge_channel.upper(), 
                             momp_channel.upper()) # get all cell features
 

print("Tracking") 

sequence.TrackCells(bidirectional=bidirectional) # track cells
sequence.PlotTracking() # plot the tracking graph
  

#Get the segmented cells list with all the cell properties, take a cell every 10 cells
root_cell_list = tuple(
    root_cell for c_idx, root_cell in enumerate(sequence.RootCells()) if c_idx %10 == 0
)


feature_names_list=[]
for feature in sequence.CellFeatureNames(): # for each cell feature 
    feature_names_list.append(feature)
    sequence.PlotCellFeatureEvolutions(root_cell_list, feature, show_figure=False) # plot feature evolution
    

#############   SAVE FEATURES   #################
    
print("Save Features ...")
features=sequence.SaveCellFeatureEvolution(feature_names_list)


# cell labeling with the corisponding unique identifier 
print("Cell Labeling")

sequence.CellLabeling(segm_channel.upper(),features,uid)


#############   DEATH TIME EVALUATION   #################

print("\nCell death Time Evaluation")

print ("Compute cell death parameters...")

death_param=CellDeath_t.EvalCellDeathTime(features)

print("\nTrajectory Processing ")
trajectories=process_traj.EvalTrack(death_param)

print("\nDone")


