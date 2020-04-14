#from task import normalization as nm_
#from extra import bg_subtraction as bg_sub
from type.sequence import sequence_t
#from analysis.death_time import CellDeath_t
#from analysis.process_trajectories import process_traj
from run_parameters import *
from model import train_new_model as tm_
import os
import numpy as np_
#import csv
#%%
if not os.path.exists("result"):
    os.makedirs("result")
    

    
if new_model.lower() == "cell_model" :
    
    tm_.Train_model(channel, cell_model)

elif new_model.lower() == "nucl_model" :
    
    tm_.Train_model(channel, nucl_model)
    
elif new_model.lower() == "cell_nucl_model" :
    
    tm_.Train_model(channel, cell_model)
    
    tm_.Train_model(channel, nucl_model)
    
           
############# sequence redaing #############
    
print("\nReading Sequence")

files= os.listdir(sequence_path)

for file in files :
    
    print(f"\nLoad ==> {file}\n")
    if not os.path.exists("output_"+str(file)):
        os.makedirs("output_"+str(file))
    if sequence_path[-1] != "/" :
        sequence_path=sequence_path+"/"
        
    file_path= sequence_path+str(file)
    
    sequence = sequence_t.FromTiffFile(
        file_path,
        acq_seq,
        from_frame=from_frame,
        to_frame=to_frame, 
        n_channels=n_channels)
    
# organize channels
    
    sequence.OrganizeFrames(acq_seq)

############# channel alignement #############

    if n_channels >1:
        print("\nRegistration")
        sequence.RegisterChannelsOn(ref_channel.upper()) 
    
    
############ contrast normalization #############
#        
#    print("\nBackground Normalization")
#    
#    #bg subtraction
#    #acq_seq2=["CHERRY","___"]
#    post_processing_fct = lambda img: bg_sub.ContrastNormalized(img,(43, 43)) # 63
#    sequence.BackgroundNormalization(acq_seq,post_processing_fct=post_processing_fct)
#    
##%%
#    #histogram equalization                                
#    post_processing_fct = lambda img: nm_.ContrastNormalized(img,percentile) # to run the contrast normalization script
#    sequence.BackgroundNormalization(acq_seq,post_processing_fct=post_processing_fct)
#    
#
#   # crop 
#    sequence.Rescale_img("yfp",(43, 43))
#      
#    
#    
##############   SEGMENTATION   #############
    
    print("\nSegmentation")
    
    sequence.SegmentCellsOnChannel( cell_channel, nucl_channel, cyto_seg)
    
    print(sequence)

#    sequence.PlotSegmentations(file) # plot the segmentation result

#%%
##############   TRACKING   #################
#
    print("\nTracking Feature Computation")
    
    intensities=sequence.ComputeCellFeatures(segm_protocol.lower(),same_channel,signal_channel,
                                           cell_sig, nucl_sig, cyto_sig,
                                           edge_channel.upper(), momp_channel.upper()) # get all cell features
    
    
    







# 
##%%
#    print("Tracking") 
#    
#    sequence.TrackCells(bidirectional=bidirectional) # track cells
#    sequence.PlotTracking() # plot the tracking graph
#      
#    
#    #Get the segmented cells list with all the cell properties, take a cell every 10 cells
#    root_cell_list = tuple(
#        root_cell for c_idx, root_cell in enumerate(sequence.RootCells()) if c_idx %1 == 0
#    )
#    
#    
#    feature_names_list=[]
#    for feature in sequence.CellFeatureNames(): # for each cell feature 
#        feature_names_list.append(feature)
#        sequence.PlotCellFeatureEvolutions(root_cell_list, feature,file, show_figure=False) # plot feature evolution
##%%  
#
##############   SAVE FEATURES   #################
#  
#    print("Save Features ...")
#    features=sequence.SaveCellFeatureEvolution(feature_names_list,file)
#
##%%
#    sequence.WriteCellFeatureEvolution(features,file)
#
#    print("Plot_line")
#    
#    #sequence.Plot_line("YFP",features)
#
##%%    
## cell labeling with the corresponding unique identifier 
#    print("Cell Labeling")
#    
#    sequence.CellLabeling(segm_channel.upper(),features,uid,file)
#    
#    
##############   DEATH TIME EVALUATION   #################
#    
##    print("\nCell death Time Evaluation")
##    
##    print ("Compute cell death parameters...")
##    
##    death_param=CellDeath_t.EvalCellDeathTime(features)
##    
##    print("\nTrajectory Processing ")
##    trajectories=process_traj.EvalTrack(death_param)
#    
#    print("\nDone")
#
##%%
#    
#from matplotlib import pyplot as pl_
#
#if not os.path.exists("output_/signal"):
#    os.makedirs("output_/signal")
##
##
##for sig in range(0,len(features[1]["signal"])):
##    pl_.figure()
##    pl_.plot(features[1]["signal"][sig])
##    pl_.title(f"root_cell 1 - frame {sig}")
##    title= "output_/signal/frame-"+str(sig)+".jpg"           
##    pl_.savefig(title) 
#    
#for cell in range(0,len(features)):
#   
#    title= "output_/signal/cell"+str(cell)+".jpg"
#    pl_.figure()
#    pl_.plot(features[cell]["signal"][0],"-x")
#    pl_.title(f"root_cell {features[cell]['uid'][0]}")           
#    pl_.savefig(title) 
#
#
