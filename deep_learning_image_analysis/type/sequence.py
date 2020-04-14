from __future__ import annotations
from type.cell import cell_t
#from type import get_properties as props_
from task import feature as ft_, segmentation as sg_
from type.frame import frame_t
from type.tracks import tracks_t
from run_parameters import segm_protocol
import imageio as io_
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

#from extra import bg_subtraction as bg_sub

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
        frame_reader = io_.get_reader(path) # read the frame
        
        assert n_channels * (to_frame + 1) <= frame_reader.get_length(), \
            f"Last frame {to_frame} outside sequence ({frame_reader.get_length()} frame(s))"

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
                print(channel)
                print(len(list_frames))
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

            for f_idx, frame in enumerate(frames):
                if frame.cells is not None:
                    self_as_str += (
                        f"    {initial_time_point+f_idx}: c {frame.cells.__len__()}\n" # time point and the number of cells in the frame  
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
                    
               
    def Rescale_img(self,channel,block):
        
        channel_name=channel.upper()
        block_half_shape = (block[0] // 2, block[1] // 2)
        for frames_ in self.frames[channel_name]:
            
#            res_img=bg_sub.BlockBasedCroppedImage(frames_.contents,block)
#            
#            frames_.contents = bg_sub(frames_.contents, block, res_img.shape)
            frames_.contents = bg_sub.ImageCroppedToEntireBlocks(frames_.contents, block)
            
    # rescale the image by a half block shape 
            frames_.contents=frames_.contents [block_half_shape[0] : (frames_.contents.shape[0] - block_half_shape[0]),block_half_shape[1] : (frames_.contents.shape[1] - block_half_shape[1])]
            frames_.contents=frames_.contents [block_half_shape[0] : (frames_.contents.shape[0] - block_half_shape[0]),block_half_shape[1] : (frames_.contents.shape[1] - block_half_shape[1])]
            
                            
#############   SEGMENTATION   #############
                
    def SegmentCellsOnChannel(self, cell_channel: None, nucl_channel: None, cyto_seg: str) -> None:
        
        """
        SegmentCellsOnChannel function
        Segment cells on each channel
        by calling the "SegmentCells" method from the "frame" script.
        """
        print("SegmentCellsOnChannel function")
        
        
        
        self.cell_channel = cell_channel.upper()
        self.nucl_channel=nucl_channel.upper()
        frames=self.channel_content
        
        
        frame_t.SegmentObject(self,frames,cell_channel, nucl_channel, cyto_seg)
       
   
        
    def CellSegmentationAt(
        self, time_point: int, binary: bool = False
    ) -> Optional[np_.ndarray]:
        
        """
        CellSegmentationAt function
        Uses the "CellSegmentation" method from the "frame" script,
        to get the cell compsosing pixels at each time point
        """
    
        if self.segm_channel is None:
            raise RuntimeError(
                "Segmentation-related function called before segmentation"
            )

        frames = self.frames[self.segm_channel] # get the segmentation channel frames
        frame = frames[time_point - frames[0].time_point] # get the frame of a specific time point

        return frame.CellSegmentation(binary=binary) # call "CellSegmentation" from "frame" script


    
    def PlotSegmentations(self,file:str, show_figure: bool = True) -> None:
        
        """
        PlotSegmentations function
        Uses the "CellSegmentationAt" to get the segmentation result at each time point.
        and "PlotSegmentations" function from "segmentation" script
        """
        
        if self.segm_channel is None:
            raise RuntimeError(
                "Segmentation-related function called before segmentation"
            )

        segmentations = []
        # get time points from "frame" script
        from_frame = self.frames[self.segm_channel][0].time_point # first time point
        to_frame = self.frames[self.segm_channel][-1].time_point # last time point
        for time_point in range(from_frame, to_frame + 1):
            segmentations.append(self.CellSegmentationAt(time_point)) # save the frame segmentation at each time point

        sg_.PlotSegmentations(file,segmentations, show_figure=show_figure) # call "PlotSegmentations" from "segmentation script"
                                                                      # to plot all segmentations 

    
    def RootCells(self):
        
        """
        RootCells function
        Gets the segmented cells with all the cell properties
        """
        
        if self.segm_channel is None:
            raise RuntimeError(
                "Segmentation-related function called before segmentation"
            )
        
        return self.frames[self.segm_channel][0].cells # returns the segmented root cells properties (at time_point=0)



#############   TRACKING   #################
       
    def TrackCells(
        self, max_dist: float = np_.inf, bidirectional: bool = True
    ) -> None:
        
        """
        TrackCells method
        The tracking could be bidirectional or not.
        If bidirectional: Track cells by finding the nearest neighbor of each current cell. 
        When the track is found, it's added to the tracking graph
        """
        
        if self.segm_channel is None:
            raise RuntimeError(
                "Segmentation-related function called before segmentation"
            )
        
        self.tracking = tracks_t()  # call the track_t class from "tracks" script
        frames = self.frames[self.segm_channel]  # get the frames of the channel used for segmentation
        
        for f_idx in range(1, frames.__len__()):
            prev_frame = frames[f_idx - 1] # previous frame
            curr_frame = frames[f_idx] # current frame
            
            # call the "CellPosition" method from "frame" script
            prev_pos = prev_frame.CellPositions() # get the previous positions
            
            
            curr_pos = curr_frame.CellPositions() # get the current position
            
            # calculate the spatial distance ( the shortest distance between prev_pos and curr_pos)
            all_dist = dt_.cdist(prev_pos, curr_pos) # calculate all distances between the previous and the current positions
                                                     # all_dist : is a matrix :all_dist [prev,curr] = distance between prev and curr 
            
            
            for curr_cell in curr_frame.cells: # for current cell in the list of segmented cells
                
                curr_uid = curr_cell.uid # get the current cell unique identifier 
                
                if len(all_dist[:, curr_uid]) != 0 :
            
                    prev_uid = np_.argmin(all_dist[:, curr_uid]) # get the previous cell unique identifier 
                                                                 # coresponding to the identifier of the lowest distance over all the forward distances
                    
                    dist= all_dist[prev_uid, curr_uid]
               
                
                #if all_dist[prev_uid, curr_uid] <= max_dist: # if the distance between the previous and the current identifiers <= max distance 
                    #print(f"distance: {all_dist[prev_uid, curr_uid]}")
                 
                
                    if dist <= max_dist:
                        
                        if bidirectional:
                            #sym=symmetric
                            sym_curr_uid = np_.argmin(all_dist[prev_uid, :]) # get the identifier of lowest distance over all the backward distances
                            
                            if sym_curr_uid == curr_uid:
                                
                                # Note: a cell has a unique next cell due to bidirectional constraint
                                prev_cell = prev_frame.cells[prev_uid] # get the previous cell
                                # add an edge on the graph between the previous and the current cell
                                self.tracking.add_edge(prev_cell, curr_cell) 
                            
    #                        else :
    #                            print("ok")
    #                            #prev_cell = prev_frame.cells[prev_uid]
    #                            self.tracking.add_edge(curr_cell, curr_cell)
                        
                        
                        else: # if not bidirectional
                            # previous cell
                            prev_cell = prev_frame.cells[prev_uid] 
                            #prev_pos=np_.array([[prev_cell.position[0]],[prev_cell.position[1]]])
                            
                            # current cell
                            curr_pos=np_.array([[curr_cell.position[0]],[curr_cell.position[1]]])
                    
                            self.tracking.add_edge(prev_cell, curr_cell)
                


    def PlotTracking(self, show_figure: bool = True) -> None:
        """
        PlotTracking method
        Plots the tracking 3D graphs, using the "Plot" function of "tracks" script
        """
        if self.tracking is None:
            raise RuntimeError("Tracking-related function called before tracking")

        self.tracking.Plot(show_figure=show_figure) # plot tracking
        


    def CellFeatureNames(self) -> tuple:
        """
        CellFeatureNames
        Get the cell feature names from the "features" dictionary 
        """
        # get the cell feature names from features dictionary (use "cell" script)
        return tuple(self.frames[self.segm_channel][0].cells[0].features.keys()) 




    def GetSignal(self, cell: None, nucl:None, cyto: None, uniq_channel:str, 
                  class_: str) -> None:
        
        intensities={}  
        
        
        if class_ == "cell" or class_ == "nucl":
            
            if class_ == "cell":
            
                type_=self.cells
            
            elif class_ == "nucl":
                
                type_=self.nuclei
        
            channel=uniq_channel
        
            content= self.channel_content[channel] # frame contents  
        
            intensities[channel]=[]
            
            
            for tp_,objects in enumerate(type_):
                
                frame=content[tp_]
                obj_signal=[]
                
                for obj in objects :
                    obj_signal.append(frame[obj.pixels])
                    
                intensities[channel].append(obj_signal)
        
        elif class_ == "cell and nucl" :
            import re
            cell_type=self.cells
            cell_channel=cell
            cell_content= self.channel_content[cell_channel]
            intensities["Cell"]=[]
            
            nucl_type=self.nuclei
            nucl_channel=nucl
            nucl_content= self.channel_content[nucl_channel]
            intensities["Nucl"]=[]
            
            for tp_,cells in enumerate(cell_type):
                
                cell_frame=cell_content[tp_]
                cell_signal=[]
                
                nucl_frame=nucl_content[tp_]
                nucl_signal=[]
                          
                for idx,cell in enumerate(cells) :
                    cell_pix=cell.pixels
                    
                    x_cell=str(cell_pix[0]).replace(" ","") 
                    x_cell.replace("\n", "")
                    y_cell=str(cell_pix[1]).replace(" ","")
                    y_cell.replace("\n","")
                    for id_,nucleus in enumerate(nucl_type[tp_]):
                #nucl_pix= nucl_type[tp_][idx].pixels#
                        
                        nucl_pix=nucleus.pixels
                        x_nucl=str(nucl_pix[0]).replace(" ","")
                        x_nucl.replace("\n", "")
                        y_nucl=str(nucl_pix[1]).replace(" ","")
                        y_nucl.replace("\n","")
                        
                        #print(x_nucl)
                        #print(x_cell)
                        #if np_.intersect1d(np_.asanyarray(nucl_pix[0]).all(), np_.asanyarray(cell_pix)[0].all()) and np_.intersect1d(np_.asanyarray(nucl_pix[1]).all(), np_.asanyarray(cell_pix[1]).all()) :
                        #print(re.search(x_nucl[1:-1],x_cell[1:-1]))
                        #print(re.findall(y_nucl,y_cell))
                        if re.findall(x_nucl[1:-1],x_cell[1:-1]) and re.findall(y_nucl[1:-1],y_cell[1:-1]):
                            print(x_nucl[1:-1])
                            print(x_cell[1:-1])
                            print(f"ok{id_}")
                            nucl_signal.append(nucl_frame[nucl_pix])
                            cell_signal.append(cell_frame[cell_pix])
                    
                intensities["Cell"].append(cell_signal)
                intensities["Nucl"].append(nucl_signal)
            
            
        
        return intensities
    
    
    def OrganizeFeatures(self,features:dict)->None:
        
        for channel_name, types_ in features.items():
        
            for type_,frames in types_.items():
                for idx,frame in enumerate(frames):
                    cells_pix=[]
                    cells=[]
                    for cell in frame: 
                        cells_pix.append(len(cell))
                        cells.append(cell)
                        
                    signal=np_.empty((max(cells_pix),len(frame)))*np_.NaN
                    
                    for col,cell in enumerate(cells) :
                        
                        signal[0:len(cell),col]=cell
                    
                    types_[type_][idx]=signal
                    
        return features
    
    
    def ComputeCellFeatures(self, method: str,same_channel:str, signal_channel:str,
                            cell_sig: None,
                            nucl_sig: None,
                            cyto_sig:None,
                            edge_channel: str, momp_channel: str) -> None:
        """
        ComputeCellFeatures function
        Computes cell features: intensity of each reporter, the fret ratio intensity,
        the MOMP location and edginess.
        """
    
        if self.cell_channel is None or self.nucl_channel is None :
            
            raise RuntimeError(
                "Segmentation-related function called before segmentation"
            )
        
        
        features={}
        
        if same_channel.lower() == "yes":
            
            channel=signal_channel.upper()
            
            if method.lower() == "cell" :
                
                cell_signal= self.GetSignal(channel,"cell")
                features["cell"]=cell_signal
            
            elif method == "nucl" :
                
                nucl_signal= self.GetSignal(channel,"nucl")
                features["nucl"]=nucl_signal
        
            elif method == "cell and nucl":
                
                signal= self.GetSignal(channel,channel,channel,None,
                                       "cell and nucl")
                features[channel]=signal
                
        #features=self.OrganizeFeatures(features)
        
        return features
    
    
    
                
#            # FRET ratio computation
#                for numerator,denominator in zip(self.frames[fret_ratio[0]],self.frames[fret_ratio[1]]):
#                    intensities[fret_ratio[0]]=numerator.contents[cell.pixels]
#                    intensities[fret_ratio[1]]=denominator.contents[cell.pixels]
#                    
#                    # take the median intensities
#                    cell.features[fret_ratio[0]] = np_.median(intensities[fret_ratio[0]])
#                    cell.features[fret_ratio[1]] = np_.median(intensities[fret_ratio[1]])
#                    #calculate the ratio
#                    cell.features[f"{fret_ratio[0]}-over-{fret_ratio[1]}"] =(cell.features[fret_ratio[0]]/
#                                 cell.features[fret_ratio[1]]) 
#                
#                for channel_name in content_dict.keys():    
#             # compute other channel intencities 
#                    if channel_name !=fret_ratio[0] and channel_name !=fret_ratio[1] and channel_name != "RFP" and channel_name != "CHERRY":
#                        intensities[channel_name]=content_dict[channel_name]
#                        cell.features[channel_name] = np_.median(intensities[channel_name])
#                 elif channel_name == "CHERRY":
#                        
#                        for channel in self.frames["YFP"]:
#                           
#                            cell_pix=channel.contents[cell.pixels]
#                                
#                            intensities["CHERRY"]=channel.contents[cell.pixels]
#                            list_int=[]
#                            step=round(len(cell_pix)/20)
#                            
#                            if step != 0:
#                                for idx in range(0,len(cell_pix),step) :
#    #                                if idx==0:
#                                    int_idx=np_.median(intensities["CHERRY"][idx:idx+round(len(cell_pix)/20)])
#                                
#    #                                print("intencity")
#    #                                print(int_idx)
#                                    
#                                    list_int.append(int_idx)
#                                
#                                cell.features["CHERRY"] = max(list_int)
#                                #print("intencity")
#                                #print(max(list_int))
#                            
#                           # cell.features["CHERRY"] = np_.median(intensities["CHERRY"])
#                       
#                        
#                    # if "RFP" channel exists 
#                    elif channel_name == "RFP" :
#                        
#                        if "momploc" in channel_treat_prot.lower():
#                            
#                            cell.features["MOMP"] = ft_.MOMPLoc(
#                                content_dict[momp_channel], content_dict["RFP"], cell.pixels) # get the MOMP reporter location
#                            
#                            
#                        elif "cyto" in channel_treat_prot.lower() : 
#                            
#                            cell.features["RFP"] = np_.percentile(content_dict["RFP"][cell.pixels], 80) # get the RFP percentile
#                    
#                   
#              # cell edge computation 
#                    
#                    cell.features["edge"] = np_.median(ft_.Edginess(content_dict[edge_channel],
#                                 cell.pixels,cell.position))
                    
                    
                    

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

        if not os.path.exists("output_"+str(file)+"/plot_features"):
            os.makedirs("output_"+str(file)+"/plot_features")  
            
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
            pl_.savefig("output_"+str(file)+"/plot_features/"+feature)
        else:
            pl_.close(figure)

        if show_figure:
            pl_.show()
       
        
        
    def SaveCellFeatureEvolution (self, feature_names,file)->list:
        
        """
        SaveCellFeatureEvolution function 
        Saves the cell features and properties 
        in a dictionary of features, then all the cell dictionaries in a list 
        """
        
        for c_idx, root_cell in enumerate(self.RootCells()):
            track = self.tracking.TrackContainingCell(root_cell)  
            
            if track is not None:
                
                cell_features={} # empty dictionary to save the cell features
                # create an empty list for each feature
                cell_features["time_points"]=[]
                cell_features["uid"]=[]
                cell_features["position"]=[]
                cell_features["pixels"]=[]
                for feature in feature_names:
                    cell_features[feature]=[]
               
                for from_cell, to_cell in nx_.dfs_edges(track, source=root_cell):
                    
                    for value in cell_features.values(): # save the feature first value
                        if len(value) == 0:
                            cell_features["time_points"].append(from_cell.time_point) # time point
                            cell_features["uid"].append(from_cell.uid) # cell unique identifier
                            cell_features["position"].append(from_cell.position) # cell centroid
                            cell_features["pixels"].append(from_cell.pixels) # pixels containing the cell
                            
                            for feature in feature_names:
                                if feature in from_cell.features:
                                    cell_features[feature].append(from_cell.features[feature])
                                else:
                                    cell_features[feature].append(0)
                    
                    # save the other feature values       
                    if to_cell.time_point not in cell_features["time_points"] : # take randomly one forward cell (in the case of unidirectional tracking
                                                                                # which is not optimal). once this problem is solved, the registration of 
                                                                                # all the features of the new cell must be done
                                                                                #(from t0 : take its mother cell features, to the end)
                                                                                
                        cell_features["time_points"].append(to_cell.time_point)
                        cell_features["uid"].append(to_cell.uid)
                        cell_features["position"].append(to_cell.position)
                        cell_features["pixels"].append(to_cell.pixels)
                        for feature in feature_names:
                            if feature not in to_cell.features:
                                if feature in from_cell.features:
                                    cell_features[feature].append(from_cell.features[feature])
                                elif feature not in from_cell.features:
                                   cell_features[feature].append(0) 
                            else:
                                cell_features[feature].append(to_cell.features[feature])
                        if cell_features not in self.feature_save:               
                            self.feature_save.append(cell_features)
        
        return self.feature_save
    
    
    def WriteCellFeatureEvolution(self, feature_list,file) :
        
        tab=[]
        # save features in csv file 
        
        fname= "output_"+str(file)+"/intensity.csv"
        
        wfile=open(fname,"w", newline='')
        
        writer=csv.writer(wfile)
        
       # writer.writerow(('YFP','cell'))
        
        for cell in range(0, len(feature_list)): 
            
           # writer.writerow(('cell', cell))
            list_int=feature_list[cell]["CHERRY"]
            
#            for i in range(0,len(list_int)):
#                writer.writerow(list_int[i])
            tab.append(list_int)
            
        writer.writerows(tab)
            
        wfile.close()
            
            
        
    def Plot_line(self,channel,features):
        if not os.path.exists("output_/line"):
            os.makedirs("output_/line")
         
                        
        for idx,frames_ in enumerate (self.frames[channel]):
            title= "output_/line/frame"+str(idx)+".jpg"
            fig, ax = pl_.subplots()
            
            ax.imshow(frames_.contents, cmap=pl_.cm.gray) 
            
            
            for cell in range(0,len(features)):
                
                orientation=features[cell]["orientation"]
                x0=features[cell]["position"][0][0]
                #print(f"x0: {x0}")
                y0=features[cell]["position"][0][1]
                #print(f"y0: {y0}")
                x1 = x0 + np_.sin(orientation) * 0.5 * (features[cell]["radius"][0])
                y1 = y0 - np_.cos(orientation) * 0.5 * (features[cell]["radius"][0])
                
                x2 = x0 + np_.sin(orientation) * 0.5 * (-features[cell]["radius"][0])
                y2 = y0 - np_.cos(orientation) * 0.5 * (-features[cell]["radius"][0])
                #print(f"x1: {x1}")
                #print(f"y1: {y1}")
                
               
                ax.plot((x0, x1[1]), (y0, y1[1]), '-r', linewidth=2.5)
                ax.plot((x0, x2[1]), (y0, y2[1]), '-c', linewidth=2.5)
                ax.plot(x0, y0, '.g', markersize=10)
    
            pl_.savefig(title)
            
    def CellLabeling(self,channel,features,uid,file):
        
        """
        CellLabeling function allows to identify cells in the original image,
        by labeling them with their cell unique identifier. 
        """

        if not os.path.exists("output_"+str(file)+"/labeled_segmentation"):
            os.makedirs("output_"+str(file)+"/labeled_segmentation")
            
        for idx,frames_ in enumerate (self.frames[channel]):
            pl_.figure(figsize=(30,30))
            pl_.imshow(frames_.contents, cmap="gray") 
            
            for cell in range(0,len(features)):
                
                if uid=="root_cell_uid": # the cell is labeled with the same uid over the diferent frames
                    
                    pl_.text(features[cell]["position"][0][0],features[cell]["position"][0][1],
                                 str(features[cell]["uid"][0]),color="red", fontsize=30)
                    
                if uid=="cell_uid": # the cell is labeled with the specfic frame uid, if the uid isn't available,
                                   # the cell is labeled with "x" 
                   if len(features[cell]["uid"])-1>= idx: 
                       pl_.text(features[cell]["position"][0][0],features[cell]["position"][0][1],
                                 str(features[cell]["uid"][idx]),color="red", fontsize=35)
                   else:
                       
                       pl_.text(features[cell]["position"][0][0],features[cell]["position"][0][1],
                             "x" ,color="red", fontsize= 35 )
            
            # save figures
            title= "output_"+str(file)+"/labeled_segmentation/frame"+str(idx)+".tif"           
            pl_.savefig(title) 
            pl_.close()
            
        frames_.ClearContents()