#!/usr/bin/python3
#-*- coding: utf-8 -*-

from __future__ import annotations
from type.cell import cell_t
from task import feature as ft_, segmentation as sg_
from type.frame import frame_t
from type.tracks import tracks_t
from run_parameters import segm_protocol, cyto_seg
import imageio as io_
import itk
import matplotlib.pyplot as pl_
import networkx as nx_
import numpy as np_
import scipy as sc_
import scipy.spatial.distance as dt_
from typing import Callable, Optional, Sequence, Tuple
from run_parameters import *
import os   
import csv
from skimage import filters
from numpy import ones,vstack
from numpy.linalg import lstsq




class sequence_t:

    def __init__(self) -> None:
        #
        self.frames = {}  # Dictionary "channel" -> list of frames
        self.channel_content={}
        self.cell_channel = None  
        self.nucl_channel = None
        self.tracking = None  # tracks
        self.save={} # dictionnary to save computed features 
        self.feature_save=[] # list to save computed features
        
    @classmethod
    
    def FromTiffFile(
        cls,
        path,
        channel_names: Sequence[str],
        from_frame: int = 0,
        to_frame: int = 999999,
        n_channels: int=1,
        
    ) -> sequence_t:
        
        """
        FromTiffFile function
        Reads the frames and split channels.
        Gets the frames properties : channel name, time point, content and shape
        """
        #
        # Channel name == '___' => discarded
        #
        instance = cls()

        for name in channel_names:
            
            if name != "___":
                name= name.upper()
                instance.frames[name] = [] # create an empty list for each channel

        ch_idx = n_channels - 1
        time_point = -1
        
        #get file path
        #frame_reader = io_.get_reader(path) # read the frame
        
        frame_reader=itk.imread(path)
        frame_reader=itk.array_from_image(frame_reader)#.astype(np_.uint8)
        
#        assert n_channels * (to_frame + 1) <= frame_reader.get_length(), \
#            f"Last frame {to_frame} outside sequence ({frame_reader.get_length()} frame(s))"

        for raw_frame in frame_reader:
            ch_idx += 1
            if ch_idx == n_channels: # if n_channels are read 
                ch_idx = 0 # channel index is reset to 0
                time_point += 1 # time point increment by 1

            if time_point < from_frame:
                continue
            elif time_point > to_frame:
                break
            
            name = channel_names[ch_idx].upper()
            
            if name != "___":
                
                print(f"Frame {name}.{time_point}")
                frame = frame_t.WithProperties(name, time_point, raw_frame) # get the frame properties using "frame" script
                
                instance.frames[name].append(frame) # save the frame properties 
        
        return instance # return an instance of the frames properties 
    
    
    def OrganizeFrames(self,channel_names: Sequence[str]):
        
        for channel in channel_names:     
            if channel != "___" :
                channel=channel.upper()
                
                list_frames=[]
                for frames_ in self.frames[channel]:
                    list_frames.append(frames_.contents)
               
                self.channel_content[channel]=list_frames
                
        return self.channel_content
    
    
    def __str__(self) -> str:
        
        """
        __str__ method
        Makes a string containing informations about the segmented frames
        (channel name, image size, time points, number of cells per frame)
        to print at the end of segmentation 
        """
        
        self_as_str = ""

        for channel, frames in self.frames.items():
            initial_time_point = frames[0].time_point
            self_as_str += (
                f"[{channel}]\n" # channel name
                f"    {frames[0].size[1]}x{frames[0].size[0]}x" # image size
                f"[{initial_time_point}..{frames[-1].time_point}]\n" # the first and the last time points
            )

            #for f_idx, frame in enumerate(frames):
            for f_idx, f_cell in enumerate (self.nuclei):
                if f_cell is not None:
                    self_as_str += (
                        f"    {initial_time_point+f_idx}: c {f_cell.__len__()}\n" # time point and the number of cells in the frame  
                    )

        return self_as_str # returns a string containing all the informations
    
    

    def RegisterChannelsOn(self, ref_channel: str) -> None:
        
        """
        RegisterChannelsOn function
        Aligns channels by calculating and correcting the between channel shift 
        Calls the "RegisterOn" method from "frame" script.
        """
        
        channel_names = tuple(self.frames.keys()) # get channel names
        ref_channel_idx = channel_names.index(ref_channel) # get the reference channel index
        other_channel_idc = set(range(channel_names.__len__())) - {ref_channel_idx} # indices of the other channels 

        ref_frames = self.frames[ref_channel] # get the reference frames 
        
        for c_idx in other_channel_idc: 
        
            floating_frames = self.frames[channel_names[c_idx]] # target channel
            for f_idx in range(ref_frames.__len__()):
                floating_frames[f_idx].RegisterOn(ref_frames[f_idx].contents) # get the between channel shift (see "frame" script)
            


    def BackgroundNormalization (self, channels,post_processing_fct: Callable = None): 
        
        """
        BackgroundNormalization function
        Normalizes the background using a post processing function (defiened on the run script)
        """
        
        if post_processing_fct is not None: 
        
            for channel_name in channels:     
                #if channel_name != "___" :
                if channel_name == "cherry":
                    channel_name=channel_name.upper()
                
                    for frames_ in self.frames[channel_name]:
                        frames_.contents= post_processing_fct(frames_.contents) 
                    
 

           
                            
#############   SEGMENTATION   #############
            
                
    def SegmentCellsOnChannel(self, cell_channel: None, nucl_channel: None, cyto_seg: str, file_name:str) -> None:
        
        """
        SegmentCellsOnChannel function
        Segment cells on each channel
        by calling the "SegmentCells" method from the "frame" script.
        """
        print("SegmentCellsOnChannel function")
        
        
        
        self.cell_channel = cell_channel.upper()
        self.nucl_channel=nucl_channel.upper()
        frames=self.channel_content
        
        
        frame_t.SegmentObject(self,frames,cell_channel, nucl_channel, cyto_seg, file_name)
       
   
        

    
    def RootCells(self):
        
        """
        RootCells function
        Gets the segmented cells with all the cell properties
        """
        
        if self.cell_channel is None:
            raise RuntimeError(
                "Segmentation-related function called before segmentation"
            )
        
        #return self.frames[self.cell_channel][0].cells # returns the segmented root cells properties (at time_point=0)
        return self.nuclei[0]
    
    


#############   TRACKING   #################
       
    def TrackCells(
        self,channel_class: str, bidirectional: bool = True, max_dist: float = np_.inf
    ) -> None:
        
        """
        TrackCells method
        The tracking could be bidirectional or not.
        If bidirectional: Track cells by finding the nearest neighbor of each current cell. 
        When the track is found, it's added to the tracking graph
        """
        if channel_class == "nucl":
            track_channel= self.nucl_channel
            type_= self.nuclei
        
        if track_channel is None:
            raise RuntimeError(
                "Segmentation-related function called before segmentation"
            )
        
        self.tracking = tracks_t()  # call the track_t class from "tracks" script
        #frames = self.frames[track_channel]  # get the frames of the track channel 
        
        for f_idx in range(1, type_.__len__()):
            prev_frame = type_[f_idx - 1] # previous frame
            curr_frame = type_[f_idx] # current frame
            
            # call the "CellPosition" method from "frame" script
                
                    
            prev_pos = frame_t.CellPositions(self,prev_frame) # get the previous positions
            
                
            curr_pos = frame_t.CellPositions(self,curr_frame) # get the current position
            
            # calculate the spatial distance ( the shortest distance between prev_pos and curr_pos)
            all_dist = dt_.cdist(prev_pos, curr_pos) # calculate all distances between the previous and the current positions
                                                     # all_dist : is a matrix :all_dist [prev,curr] = distance between prev and curr 
            
            
            for curr_cell in type_[f_idx]: # for current cell in the list of segmented cells
                
                curr_uid = curr_cell.uid # get the current cell unique identifier 
                
                if len(all_dist[:, curr_uid]) != 0 :
            
                    prev_uid = np_.argmin(all_dist[:, curr_uid]) # get the previous cell unique identifier 
                                                                 # coresponding to the identifier of the lowest distance over all the forward distances
                    
                    dist= all_dist[prev_uid, curr_uid]
               
                
                    if dist <= max_dist:
                        
                        if bidirectional:
                            #sym=symmetric
                            sym_curr_uid = np_.argmin(all_dist[prev_uid, :]) # get the identifier of lowest distance over all the backward distances
                            
                            if sym_curr_uid == curr_uid:
                                
                                # Note: a cell has a unique next cell due to bidirectional constraint
                                prev_cell = type_[f_idx-1][prev_uid] # get the previous cell
                                # add an edge on the graph between the previous and the current cell
                                self.tracking.add_edge(prev_cell, curr_cell) 
                            
                        
                        
                        else: # if not bidirectional
                            # previous cell
                            prev_cell = type_[f_idx-1][prev_uid] 
                            
                            # current cell
                            curr_pos=np_.array([[curr_cell.position[0]],[curr_cell.position[1]]])
                    
                            self.tracking.add_edge(prev_cell, curr_cell)
                


    def PlotTracking(self,file:str, show_figure: bool = True) -> None:
        """
        PlotTracking method
        Plots the tracking 3D graphs, using the "Plot" function of "tracks" script
        """
        if self.tracking is None:
            raise RuntimeError("Tracking-related function called before tracking")

        self.tracking.Plot(file ,show_figure=show_figure) # plot tracking
        


    def CellFeatureNames(self) -> tuple:
        """
        CellFeatureNames
        Get the cell feature names from the "features" dictionary 
        """
        # get the cell feature names from features dictionary (use "cell" script)
        #return tuple(self.frames[self.segm_channel][0].cells[0].features.keys()) 
        return tuple(self.cells[0][0].features.keys())
   
    
    
    def OrganizeFeatures(self,features:dict)->None:
        
        #for type_, channels in features.items():
        
        for type_,root_cells in features.items():
            for idx, root_cell  in enumerate(root_cells):
                if len(root_cell)>0:
                    cells_pix=[]
                    cells=[]
                    for cell in root_cell: 
                        cells_pix.append(len(cell))
                        cells.append(cell)
                        
                    signal=np_.empty((max(cells_pix),len(root_cell)))*np_.NaN
                    
                    for col,cell in enumerate(cells) :
                        
                        signal[0:len(cell),col]=cell
                    
                    root_cells[idx]=signal
                    
                elif len(root_cell)==0:
                    root_cells[idx]= 1*np_.NaN
                    
        return features
    
    
    
    def ComputeCellFeatures(self, method: str,same_channels:None, feature_channel:str,
                            cell_sig: None,
                            nucl_sig: None,
                            cyto_sig:None)-> None:
        
        
    
        if self.cell_channel is None or self.nucl_channel is None :
            
            raise RuntimeError(
                "Segmentation-related function called before segmentation"
            )
        
        intensities={}
        
        if method.lower() == "cell" :
            for channel in cell_sig:
                channel= channel.upper()
                signal, props= GetUniSignal(self,channel,feature_channel,"cell")
                intensities[channel]=self.OrganizeFeatures(signal)
                #intensities[channel]=signal
        
        elif method == "nucl" :
            for channel in nucl_sig:
                channel= channel.upper()
                nucl_signal,props= GetUniSignal(self,channel,feature_channel,"nucl")
                intensities[channel]=self.OrganizeFeatures(signal)
    
        elif method == "cell and nucl":
            
            if same_channels != None:
                
                for channel in same_channels:
                    channel= channel.upper()
#                    signal,props= signal_.GetMultiSignals(self,channel,channel,channel,feature_channel,
#                                           "cell and nucl")
                    signal,props= GetMultiSignals(self,channel,channel,channel,feature_channel,
                                           "cell and nucl")
                    intensities[channel]=self.OrganizeFeatures(signal)
                    #intensities[channel]=signal
                    
            elif same_channels == None:
                
                if len(cell_sig)==1 and len(nucl_sig)==1:
                    
                    cell_sig= cell_sig[0].upper()
                    nucl_sig= nucl_sig[0].upper()
                    
                    if cyto_sig==None :
                        
                        signal,props= GetMultiSignals(self,cell_sig,nucl_sig,None,None,
                                               "cell and nucl")
                        intensities=self.OrganizeFeatures(signal)
                        #intensities=signal
                        
                    elif cyto_sig != None and len(cyto_sig)==1:
                        
                        signal,props= GetMultiSignals(self,cell_sig,nucl_sig,cyto_sig,None,
                                               "cell and nucl")
                        
                        intensities=self.OrganizeFeatures(signal)
                        #intensities=signal
                        
                elif len(cell_sig) >1 or len(nucl_sig)>1:
                    
                    for channel in cell_sig:
                        if channel in nucl_sig:
                           
                            if cyto_sig == None:
                                channel= channel.upper()
                                signal,props= GetMultiSignals(self,channel,channel,None,channel,
                                               "cell and nucl")
                                
                                intensities[channel]=self.OrganizeFeatures(signal)
                                #intensities[channel]=signal
                            
                            elif cyto_sig != None and channel in cyto_sig:
                                channel= channel.upper()
                                signal,props= GetMultiSignals(self,channel,channel,channel,None,
                                               "cell and nucl")
                                intensities[channel]=self.OrganizeFeatures(signal)
                                #intensities=signal
                        else:
                            channel=channel.upper()
                            cell_signal, props= GetUniSignal(self,channel,feature_channel,"cell")
                            
                            intensities[channel]=self.OrganizeFeatures(signal)
                            #intensities[channel]=cell_signal
                    
                    for channel in nucl_sig:
                        if channel not in cell_sig:
                            channel=channel.upper()
                            nucl_signal, props= GetUniSignal(self,channel,feature_channel,"nucl")
                            intensities[channel]=self.OrganizeFeatures(nucl_signal)
                            #intensities[channel]=nucl_signal
                            
                    for channel in cyto_sig:
                        if channel not in cell_sig:
                            channel=channel.upper()
                            cyto_signal, props= GetUniSignal(self,channel,feature_channel,"nucl")
                            intensities[channel]=self.OrganizeFeatures(cyto_signal)
                            #intensities[channel]=cyto_signal
                    
        #signal=self.OrganizeFeatures(signal)
        
        
        return intensities,props
    
    
    
   
    def CalculSignal(self,file:str, intensity: dict, 
                     numerator:str, denominator: str, num_struct:str, denom_struct:str,
                     mean: bool, median: bool, ratio: bool):
        """
        This fuction computes statistics operations on the extracted signal
        """
        
        calc_sig={}
        

        
        for channel, types in intensity.items():
            for type_,frames in types.items():
                cell_num=[]
                
                for idx, frame in enumerate(frames):
                    if type(frame) != float :
                        cell_num.append(frame.shape[1])
                       

        cell_num=max(cell_num)
        frame_num=len(frames)
        
        
        for channel, types in intensity.items():
            calc_sig[channel]={}
            for type_,frames in types.items():
                calc_sig[channel][type_]={}
                med=np_.empty((frame_num,cell_num))*np_.NaN
                avg=np_.empty((frame_num,cell_num))*np_.NaN
                
                
                if median== True and mean ==False or mean== True and median == False:
                    if median==True :
                        operation= "median"
                        calc_sig[channel][type_][operation]={}
                        
                     
                        for idx, frame in enumerate(frames): 
                            for col in range(0,frame.shape[1]):
                                
                                signal=frame[:,col]
                                signal=signal[~ np_.isnan(signal)]
                                
                                med[idx,col]= np_.median(signal)
                        sig=med
                                
                    elif mean==True:
                        operation="mean"
                        calc_sig[channel][type_][operation]={}
                        
                     
                        for idx, frame in enumerate(frames): 
                            for col in range(0,frame.shape[1]):
                        
                                signal=frame[:,col]
                                signal=signal[~ np_.isnan(signal)]
                                
                                avg[idx,col]= np_.mean(signal)
                                
                        sig=avg
                    
                    
                    
                    
                    calc_sig[channel][type_][operation]=sig
                    
                    if not os.path.exists("output/"+str(file)+"/signal/"+str(operation)):
                        os.makedirs("output/"+str(file)+"/signal/"+str(operation))
                    
                    fname= "output/"+str(file)+"/signal/"+str(operation)+"/"+str(type_)+"_"+str(operation)+".csv"
            
                    np_.savetxt(fname, sig, delimiter=",")
                
                elif median == True and mean == True :
                    
                    if not os.path.exists("output/"+str(file)+"/signal/median_intensity"):
                        os.makedirs("output/"+str(file)+"/signal/median_intensity")
                                
                    if not os.path.exists("output/"+str(file)+"/signal/mean_intensity"):
                        os.makedirs("output/"+str(file)+"/signal/mean_intensity") 
                        
                    calc_sig[channel][type_]["median"]={}
                    calc_sig[channel][type_]["mean"]={}
                    
                           
                         
                    for idx, frame in enumerate(frames):
                        if type(frame) != float :
                            for col in range(0,frame.shape[1]):
                        
                                signal=frame[:,col]
                                signal=signal[~ np_.isnan(signal)]
                                
                                med[idx,col]=np_.median(signal)
                                avg[idx,col]= np_.mean(signal)
                                     
                            calc_sig[channel][type_]["median"]=med
                            calc_sig[channel][type_]["mean"]=avg
                            
                            
                                    
                            fname1= "output/"+str(file)+"/signal/median_intensity/"+str(type_)+"_median.csv"
                            np_.savetxt(fname1, med, delimiter=",")
                            
                            fname2= "output/"+str(file)+"/signal/mean_intensity/"+str(type_)+"_mean.csv"
                            np_.savetxt(fname2, avg, delimiter=",")
                        
                
                            
                            
        if ratio== True :
           
            
            if median==True and mean==False :
                
                ratio_mat=calc_sig[numerator.upper()][num_struct.lower()]["median"]/calc_sig[denominator.upper()][denom_struct.lower()]["median"]
                calc_sig["median_ratio"]=ratio_mat
                fname= "output/"+str(file)+"/signal/median_ratio_.csv"
                np_.savetxt(fname, ratio_mat, delimiter=",")
                
            elif mean == True and median==False:
                ratio_mat=calc_sig[numerator.upper()][num_struct.lower()]["mean"]/calc_sig[denominator.upper()][denom_struct.lower()]["mean"]
                calc_sig["mean_ratio"]=ratio_mat
                fname= "output/"+str(file)+"/signal/mean_ratio_.csv"
                np_.savetxt(fname, ratio_mat, delimiter=",")
                
            elif median== True and mean == True :
                ratio_med=calc_sig[numerator.upper()][num_struct.lower()]["median"]/calc_sig[denominator.upper()][denom_struct.lower()]["median"]
                calc_sig["median_ratio"]=ratio_med
                fname= "output/"+str(file)+"/signal/median_ratio_.csv"
                np_.savetxt(fname, ratio_med, delimiter=",")
                
                ratio_avg=calc_sig[numerator.upper()][num_struct.lower()]["mean"]/calc_sig[denominator.upper()][denom_struct.lower()]["mean"]
                calc_sig["mean_ratio"]=ratio_avg
                fname1= "output/"+str(file)+"/signal/mean_ratio_.csv"
                np_.savetxt(fname1, ratio_avg, delimiter=",")
            
        return calc_sig
            
            
                    
                    
                    

    def CellFeatureEvolution(self, root_cell: cell_t, feature: str) -> list:
        
        """
        CellFeatureEvolution function
        Calculates the feature evolution of each cell between the previous and the current position.
        Use the "TrackContainingCell" method of the "tarcks" script. 
        """
    
        if self.tracking is None:
            raise RuntimeError("Tracking-related function called before tracking")
        # self.segm_channel is necessarily not None
        
        evolution = []
       
        
        track = self.tracking.TrackContainingCell(root_cell) # call the tracking method from "tracks" script
        if track is not None:
            current_piece = []
            for from_cell, to_cell in nx_.dfs_edges(track, source=root_cell): # get the current and the next cell positions idx
                
                if current_piece.__len__() == 0:
                    if feature in from_cell.features:
                        current_piece.append(
                            (from_cell.time_point, from_cell.features[feature]) # save the current cell position time point and features
                        )
                    else:
                        current_piece.append(
                            (from_cell.time_point, 0) # save the current cell position time point and features
                        )
                    
                if feature not in to_cell.features:
                    if feature in from_cell.features:
                        current_piece.append((to_cell.time_point, from_cell.features[feature]))
                    elif feature not in from_cell.features:
                        current_piece.append((to_cell.time_point, 0))
                else:
                    current_piece.append((to_cell.time_point, to_cell.features[feature])) # save the next cell position time point and features
                
                if track.out_degree(to_cell) != 1: # if the number of edges pointing out of the node is # 1 
                    evolution.append(current_piece) # save the current piece (time_point,feature)
                    current_piece = []
        
        return evolution
    
    

    def PlotCellFeatureEvolutions(
        self,
        cell_list: Sequence,
        feature: str,
        file: str,
        show_figure: bool = True,
    ) -> None:
        
        """
        This function plots the cell feature evolutions 
        calculated by the "CellFeatureEvolution" function
        """
        
        figure = pl_.figure() 
        axes = figure.gca() # get the current axes
        axes.set_title(feature) # set a title  
        axes.set_xlabel("time points") # set the x label
        axes.set_ylabel("feature value") # set the y label
        plots = []
        labels = []
        colors = "bgrcmyk"


        for root_cell in cell_list:
            color_idx = root_cell.uid % colors.__len__()
            subplot = None

            for piece in self.CellFeatureEvolution(root_cell, feature): # get the feature evolution
                if not (isinstance(piece[0][1], int) or isinstance(piece[0][1], float)): # if the feature value isn't "int" or "float" type
                    break

                time_points = []
                feature_values = [] 
                for time_point, feature_value in piece: # get the time point and the feature value 
                    time_points.append(time_point) # save the time point
                    feature_values.append(feature_value) # save the feature value
                
                subplot = axes.plot(
                    time_points, feature_values, colors[color_idx] + "-x"
                )[0] # plot f(time_points, feature_values)
                
            if subplot is not None:
                plots.append(subplot)
                labels.append(f"root_cell {root_cell.uid}") # get uid labels

        if plots.__len__() > 0: # if "plots" list isn't empty
            axes.legend(handles=plots, labels=labels, loc='lower right') # place the legend on the axes
            # save plot 
            pl_.savefig(str(file)+"/plot_features/"+feature)
        else:
            pl_.close(figure)

        if show_figure:
            pl_.show()
       
        
        
    
    
    def WriteCellFeatureEvolution(self, feature_list,file,type_,idx) :
        
        
        # save features in csv file 
        
        if not os.path.exists("output/"+str(file)+"/signal/total_intencity/"+str(type_)):
            os.makedirs("output/"+str(file)+"/signal/total_intencity/"+str(type_))
            
        fname= "output/"+str(file)+"/signal/total_intencity/"+str(type_)+"/frame"+str(idx)+"intensity.csv"
        
        
        np_.savetxt(fname, feature_list, delimiter=",", fmt='%s')

            
            
        
        
        
    def CellLabeling(self,channel,singnal,uid,file, features,intensity):
    
        """
        CellLabeling function allows to identify cells in the original image,
        by labeling them with their cell unique identifier. 
        """
    
        if self.cell_channel != None :
            
            name="segmentation_map"
            
            if not os.path.exists("output/"+str(file)+"/"+name):
                os.makedirs("output/"+str(file)+"/"+name)
            
            
        #        for idx,frame in enumerate (self.channel_content[channel]):
        #            
        #            if idx % 10 == 0 :
        #                pl_.figure(figsize=(30,30))
        #                pl_.imshow(frame, cmap="gray")
        #    
        #                for cell in range(0,len(features["nucl"]["uid"])):
        #    
        #                    if uid=="root_cell_uid": # the cell is labeled with the same uid over the diferent frames
        #                        if len(features["nucl"]["uid"][cell])>0 and idx < len(features["nucl"]["position"][cell]):
        #                            pl_.text(features["nucl"]["position"][cell][idx][0],features["nucl"]["position"][cell][idx][1],
        #                                         str(features["nucl"]["uid"][cell][0]),color="red", fontsize=40)
        #    
        #                    elif uid=="cell_uid": # the cell is labeled with the specfic frame uid, if the uid isn't available,
        #                                       # the cell is labeled with "x"
        #                       if len(features["nucl"]["position"][cell])-1>= idx:
        #                           pl_.text(features["nucl"]["position"][cell][0][0],features["nucl"]["position"][cell][0][1],
        #                                     str(features["nucl"]["uid"][cell][idx]),color="red", fontsize=40)
        #                       else:
        #    
        #                           pl_.text(features["nucl"]["position"][cell][0][0],features["nucl"]["position"][cell][0][1],
        #                                 "x" ,color="red", fontsize= 40 )
       
        idx=0       
        frame= self.channel_content[channel][idx]
        
        pl_.figure(figsize=(30,30))
        pl_.imshow(frame, cmap="gray")
        
        
        for cell in range(0,len(features["nucl"]["uid"])):
            if len(features["nucl"]["uid"][cell])>0 :
                pl_.text(features["nucl"]["position"][cell][idx][0],features["nucl"]["position"][cell][idx][1],
                             str(features["nucl"]["uid"][cell][0]),color="red", fontsize=40)

       

        # save figures
       
        path= "output/"+str(file)+"/"+name+"/frame"+str(idx)+".jpg"           
        pl_.savefig(path) 
        pl_.close()

        
        
        
        
        
def GetMultiSignals(self,cell: None, nucl:None, cyto: None, features_channel:str, 
                  class_: str):
        
    intensities={}  
    features={}
        
    cell_type=self.cells
    cell_channel=cell
    cell_content= self.channel_content[cell_channel]
    intensities["cell"]=[]
    #intensities["Cell"][cell_channel]=[]
    features["cell"]={}
    features["cell"]["time_points"]=[]
    features["cell"]["pixels"]=[]
    features["cell"]["uid"]=[]
    features["cell"]["position"]=[]
    features["cell"]["area"]=[]
    
    nucl_type=self.nuclei
    nucl_channel=nucl
    nucl_content= self.channel_content[nucl_channel]
    intensities["nucl"]=[]
    #intensities["Nucl"][nucl_channel]=[]
    features["nucl"]={}
    features["nucl"]["time_points"]=[]
    features["nucl"]["pixels"]=[]
    features["nucl"]["uid"]=[]
    features["nucl"]["position"]=[]
    features["nucl"]["area"]=[]
    
    if cyto != None :
        
        cyto_type=self.cyto
        cyto_channel=cyto
        cyto_content= self.channel_content[cyto_channel]
        intensities["cyto"]=[]
        #intensities["Cyto"][cyto_channel]=[]
        features["cyto"]={}
        features["cyto"]["time_points"]=[]
        features["cyto"]["pixels"]=[]
        features["cyto"]["uid"]=[]
        features["cyto"]["position"]=[]
        features["cyto"]["area"]=[]

        
    for c_idx, root_cell in enumerate(self.RootCells()):
        
        track = self.tracking.TrackContainingCell(root_cell)  
        
        cell_signal=[]
        cell_tp=[]
        cell_pixels=[]
        cell_uid=[]
        cell_position=[]
        cell_area=[]
        
        nucl_signal=[]
        nucl_tp=[]
        nucl_pixels=[]
        nucl_uid=[]
        nucl_position=[]
        nucl_area=[]
        
        if cyto_seg == "yes" :
            cyto_signal=[]
            cyto_tp=[]
            cyto_pixels=[]
            cyto_uid=[]
            cyto_position=[]
            cyto_area=[]
        
        if track is not None:
            
            for from_nucl, to_nucl in nx_.dfs_edges(track, source=root_cell):
                
                if track.out_degree[to_nucl] < 3:
                    
                 
                    from_tp=from_nucl.time_point
                    to_tp=to_nucl.time_point
                    
                    from_cell_frame=cell_content[from_tp]
                    from_nucl_frame=nucl_content[from_tp]
                    
                    to_cell_frame=cell_content[to_tp]
                    to_nucl_frame=nucl_content[to_tp]
                    
                    if from_tp == 0:
                        
                        for from_idx, from_cell in enumerate(cell_type[from_tp]) :
                            
                            from_cell_pix=set(zip(*from_cell.pixels))
                            from_nucl_pix=set(zip(*from_nucl.pixels))
                        
                                
                            if len(from_nucl_pix.intersection(from_cell_pix))>0 and from_nucl.features["area"]>50 and from_cell.features["area"]>100:
                                cell_signal.append(from_cell_frame[from_cell.pixels])
                                nucl_signal.append(from_nucl_frame[from_nucl.pixels])
                                
                                cell_pixels.append(from_cell.pixels)
                                cell_tp.append(from_tp)
                                cell_uid.append(from_cell.uid)
                                cell_position.append(from_cell.position)
                                cell_area.append(from_cell.features["area"])
                                
                                
                                nucl_pixels.append(from_nucl.pixels)
                                nucl_tp.append(from_tp)
                                nucl_uid.append(from_nucl.uid)
                                nucl_position.append(from_nucl.position)
                                nucl_area.append(from_nucl.features["area"])
                                
                                
                                if cyto_seg == "yes" :
                                    
                                    from_cyto_frame=cyto_content[from_tp]
                                    
                                    for id_,cyto in enumerate(cyto_type[from_tp]):
                                        cyto_pix= set(zip(*cyto.pixels))
                                    
                                        if len(from_cell_pix.intersection(cyto_pix))>0 and cyto.features["area"]>50:
                                    
                                            cyto_signal.append(from_cyto_frame[cyto.pixels])
                                            
                                            cyto_pixels.append(cyto.pixels)
                                            cyto_tp.append(from_tp)
                                            cyto_uid.append(cyto.uid)
                                            cyto_position.append(cyto.position)
                                            cyto_area.append(cyto.features["area"])
                                            
                                            
                                            
                        for to_idx, to_cell in enumerate(cell_type[to_tp]) :
                                
                            to_cell_pix=set(zip(*to_cell.pixels))
                            to_nucl_pix=set(zip(*to_nucl.pixels))
                            
                                    
                            if len(to_nucl_pix.intersection(to_cell_pix))>0 and to_nucl.features["area"]>50 and to_cell.features["area"]>100:
                                
                                cell_signal.append(to_cell_frame[to_cell.pixels])
                                nucl_signal.append(to_nucl_frame[from_nucl.pixels])
                                
                                cell_pixels.append(to_cell.pixels)
                                cell_tp.append(to_tp)
                                cell_uid.append(to_cell.uid)
                                cell_position.append(to_cell.position)
                                cell_area.append(to_cell.features["area"])
                               
                                
                                nucl_pixels.append(to_nucl.pixels)
                                nucl_tp.append(to_tp)
                                nucl_uid.append(to_nucl.uid)
                                nucl_position.append(to_nucl.position)
                                nucl_area.append(to_nucl.features["area"])
                                
                                
                                if cyto_seg == "yes" :
                                    
                                    to_cyto_frame=cyto_content[to_tp]
                                    
                                    for id_,cyto in enumerate(cyto_type[to_tp]):
                                        cyto_pix= set(zip(*cyto.pixels))
                                    
                                        if len(to_cell_pix.intersection(cyto_pix))>0 and cyto.features["area"]>50:
                                    
                                            cyto_signal.append(to_cyto_frame[cyto.pixels])
                                            
                                            cyto_pixels.append(cyto.pixels)
                                            cyto_tp.append(from_tp)
                                            cyto_uid.append(cyto.uid)
                                            cyto_position.append(cyto.position)
                                            cyto_area.append(cyto.features["area"])
                                            
                                            
                              
                                        
                    elif from_tp != 0: 
                        
                        for to_idx, to_cell in enumerate(cell_type[to_tp]) :
                                
                            to_cell_pix=set(zip(*to_cell.pixels))
                            to_nucl_pix=set(zip(*to_nucl.pixels))
                            
                            if len(to_nucl_pix.intersection(to_cell_pix))>0 and to_nucl.features["area"]>50 and to_cell.features["area"]>100:
                                cell_signal.append(to_cell_frame[to_cell.pixels])
                                nucl_signal.append(to_nucl_frame[to_nucl.pixels])
                                
                                cell_pixels.append(to_cell.pixels)
                                cell_tp.append(to_tp)
                                cell_uid.append(to_cell.uid)
                                cell_position.append(to_cell.position)
                                cell_area.append(to_cell.features["area"])
                                
                                
                                nucl_pixels.append(to_nucl.pixels)
                                nucl_tp.append(to_tp)
                                nucl_uid.append(to_nucl.uid)
                                nucl_position.append(to_nucl.position)
                                nucl_area.append(to_nucl.features["area"])
                                
                        
                                if cyto_seg == "yes" :
                                    
                                    to_cyto_frame=cyto_content[to_tp]
                                    for id_,cyto in enumerate(cyto_type[to_tp]):
                                        cyto_pix= set(zip(*cyto.pixels))
                                    
                                        if len(to_cell_pix.intersection(cyto_pix))>0 and cyto.features["area"]>50:
                                    
                                            cyto_signal.append(to_cyto_frame[cyto.pixels])
                                
                                            cyto_pixels.append(cyto.pixels)
                                            cyto_tp.append(from_tp)
                                            cyto_uid.append(cyto.uid)
                                            cyto_position.append(cyto.position)
                                            cyto_area.append(cyto.features["area"])
                                            
                
            
            
            
            
            
         #cell
        features["cell"]["pixels"].append(cell_pixels)
        features["cell"]["time_points"].append(cell_tp)
        features["cell"]["uid"].append(cell_uid)
        features["cell"]["position"].append(cell_position)
        features["cell"]["area"].append(cell_area)
         
        #nucl
        features["nucl"]["pixels"].append(nucl_pixels)
        features["nucl"]["time_points"].append(nucl_tp)
        features["nucl"]["uid"].append(nucl_uid)
        features["nucl"]["position"].append(nucl_position)
        features["nucl"]["area"].append(nucl_area)
                    
                    
        intensities["cell"].append(cell_signal)
        intensities["nucl"].append(nucl_signal)
        
        if cyto != None :
            intensities["cyto"].append(cyto_signal)
            
            features["cyto"]["pixels"].append(cyto_pixels)
            features["cyto"]["time_points"].append(cyto_tp)
            features["cyto"]["uid"].append(cyto_uid)
            features["cyto"]["position"].append(cyto_position)
            features["cyto"]["area"].append(cyto_area)        
        
            
            
    return intensities,features
            
            

def GetUniSignal(self, channel:str, features_channel:str, class_: str) -> None:
        
    intensities={}  
    features={}
    
    
    if class_ == "cell": 
        
        class_=class_.lower()
        intensities[class_]={}
        type_=self.cells
            
        
    elif class_ == "nucl":
        class_=class_.lower()
        intensities[class_]={}
        type_=self.nuclei
            
    elif class_ == "cyto":
        class_=class_.lower()
        intensities[class_]={}
        type_=self.cyto
        
        
    if channel != "___":
        channel=channel.upper()
        content= self.channel_content[channel] # frame contents  
    
        intensities[class_.upper()][channel]=[]
        
        if channel== features_channel.upper():
            features[class_]={}
            features[class_]["pixels"]=[]
            features[class_]["uid"]=[]
            features[class_]["position"]=[]
            features[class_]["area"]=[]
            
        for c_idx, root_cell in enumerate(self.RootCells()):
        
            track = self.tracking.TrackContainingCell(root_cell) 
            
      
            if track is not None:
            
                for from_tp, to_tp in nx_.dfs_edges(track, source=root_cell):
                
                    frame=content[from_tp]
                    obj_signal=[]
                    if channel== features_channel.upper():
                        pixels=[]
                        uid=[]
                        position=[]
                        area=[]
                    
                    
                        
                    for from_idx, obj in enumerate(type_[from_tp]) :
                        
                        obj_signal.append(frame[obj.pixels])
                        
                        if channel== features_channel.upper():
                            pixels.append(obj.pixels)
                            uid.append(obj.uid)
                            position.append(obj.position)
                            area.append(obj.features["area"])
                            
                    intensities[class_][channel].append(obj_signal)
                    if channel== features_channel.upper():
                        features[class_]["pixels"].append(pixels)
                        features[class_]["uid"].append(uid)
                        features[class_]["position"].append(position)
                        features[class_]["area"].append(area)
                
    return intensities, features 





    





