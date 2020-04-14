from __future__ import annotations
from type.cell import cell_t
from task import feature as ft_, segmentation as sg_
from type.frame import frame_t
from type.tracks import tracks_t
from run_parameters import segm_protocol
import imageio as io_
import matplotlib.pyplot as pl_
import networkx as nx_
import numpy as np_
import scipy.spatial.distance as dt_
from typing import Callable, Optional, Sequence, Tuple
from run_parameters import *
import os


class sequence_t:

    def __init__(self) -> None:
        #
        self.frames = {}  # Dictionary "channel" -> list of frames
        self.segm_channel = None  # Channel used for segmentation
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
                instance.frames[name] = [] # create an empty list for each frame

        ch_idx = n_channels - 1
        time_point = -1
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
                if channel_name != "___" :
                    channel_name=channel_name.upper()

                    for frames_ in self.frames[channel_name]:
                        frames_.contents= post_processing_fct(frames_.contents)




#############   SEGMENTATION   #############

    def SegmentCellsOnChannel(self, channel: str,show_sg_steps: bool = False,n_frames: int=0) -> None:

        """
        SegmentCellsOnChannel function
        Segment cells on each channel
        by calling the "SegmentCells" method from the "frame" script.
        """

        self.segm_channel = channel

        idx=0
        for frame in self.frames[channel]:

            frame.SegmentCells(segm_protocol.lower(),show_sg_steps,idx,n_frames) # cell "SegmentCells" (see "frame" script)
            idx+=1


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



    def PlotSegmentations(self, show_figure: bool = True) -> None:

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

        sg_.PlotSegmentations(segmentations, show_figure=show_figure) # call "PlotSegmentations" from "segmentation script"
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
        self, max_dist: float = np_.inf, bidirectional: bool = False
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

            # calculate the spatial distance ( the shortest distence between prev_pos and curr_pos)
            all_dist = dt_.cdist(prev_pos, curr_pos) # calculate all distances between the previous and the current positions
                                                     # all_dist : is a matrix :all_dist [prev,curr] = distance between prev and curr


            for curr_cell in curr_frame.cells: # for current cell in the list of segmented cells
                curr_uid = curr_cell.uid # get the current cell unique identifier
                prev_uid = np_.argmin(all_dist[:, curr_uid]) # get the previous cell unique identifier
                                                             # coresponding to the identifier of the lowest distance over all the forward distances
                if all_dist[prev_uid, curr_uid] <= max_dist: # if the distance between the previous and the current identifiers <= max distance
                    #print(f"distance: {all_dist[prev_uid, curr_uid]}")
                    if bidirectional:
                        # sym=symmetric
                        sym_curr_uid = np_.argmin(all_dist[prev_uid, :]) # get the identifier of lowest distance over all the backward distances
                        if sym_curr_uid == curr_uid:
                            # Note: a cell has a unique next cell due to bidirectional constraint
                            prev_cell = prev_frame.cells[prev_uid] # get the previous cell
                            # add an edge on the graph between the previous and the current cell
                            self.tracking.add_edge(prev_cell, curr_cell)

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




    def ComputeCellFeatures(self, channels: Tuple[str],channel_treat_prot: str,fret_ratio:list,
                            edge_channel: str, momp_channel: str) -> None:
        """
        ComputeCellFeatures function
        Computes cell features: intensity of each reporter, the fret ratio intensity,
        the MOMP location and edginess.
        """

        if self.segm_channel is None:

            raise RuntimeError(
                "Segmentation-related function called before segmentation"
            )


        content_dict={} # frame contents dictionary
        intensities={}  # cell intensities dictionary

        fret_ratio[0]= fret_ratio[0].upper()
        fret_ratio[1]= fret_ratio[1].upper()

        intencities[fret_ratio[0]]=[]
        intencities[fret_ratio[1]]=[]
        # organize the frame contents in a dictionary
        for channel_name in channels:
            if channel_name != "___" :
                channel_name=channel_name.upper()
                content_dict[channel_name]=[]
                for frames_ in self.frames[channel_name]:
                    content_dict[channel_name].append(frames_.contents)



        for segm_frames in self.frames[self.segm_channel]:

            for cell in segm_frames.cells:

            # FRET ratio computation
                for numerator,denominator in zip(self.frames[fret_ratio[0]],self.frames[fret_ratio[1]]):
                    for idx, frame in enumerate(numerator.contents):
                        intensities[fret_ratio[0]].append(frame[cell.pixels])
                        intensities[fret_ratio[1]].append(denominator.contents[idx][cell.pixels])

                    # take the median intensities
                        for idx, intensity in enumerate(intensities[fret_ratio[0]]):
                            cell.features[fret_ratio[0]] = np_.median(intensity)
                        for idx, intensity in enumerate(intensities[fret_ratio[1]]):
                            cell.features[fret_ratio[1]] = np_.median(intensity)
                        #calculate the ratio
                        cell.features[f"{fret_ratio[0]}-over-{fret_ratio[1]}"] =(cell.features[fret_ratio[0]]/
                                     cell.features[fret_ratio[1]])

                for channel_name,frames in content_dict.items():
            # compute other channel intencities
                    if channel_name !=fret_ratio[0] and channel_name !=fret_ratio[1] and channel_name != "RFP":
                        intensities[channel_name]=[]
                        for frame in frames:
                            intensities[channel_name].append(frame.[cell.pixels])
                            cell.features[channel_name] = np_.median(frame.[cell.pixels])

                    #if "RFP" channel exists
                    elif channel_name == "RFP" :

                        if "momploc" in channel_treat_prot.lower():
                            for frame in content_dict[momp_channel]:
                                cell.features["MOMP"] = ft_.MOMPLoc(
                                    frame, content_dict["RFP"], cell.pixels) # get the MOMP reporter location


                        elif "cyto" in channel_treat_prot.lower() :
                            for frame in content_dict["RFP"]:
                                cell.features["RFP"] = np_.percentile(frame[cell.pixels], 80) # get the RFP percentile


              # cell edge computation
                    for frame in content_dict[edge_channel]:
                        cell.features["edge"] = np_.median(ft_.Edginess(frame,
                                     cell.pixels,cell.position))




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
                    current_piece.append(
                        (from_cell.time_point, from_cell.features[feature]) # save the current cell position time point and features
                    )
                current_piece.append((to_cell.time_point, to_cell.features[feature])) # save the next cell position time point and features

                if track.out_degree(to_cell) != 1: # if the number of edges pointing out of the node is # 1
                    evolution.append(current_piece) # save the current piece (time_point,feature)
                    current_piece = []

        return evolution



    def PlotCellFeatureEvolutions(
        self,
        cell_list: Sequence,
        feature: str,
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

        if not os.path.exists("outputs/plot_features"):
            os.makedirs("outputs/plot_features")

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
            pl_.savefig("outputs/plot_features/"+feature)
        else:
            pl_.close(figure)

        if show_figure:
            pl_.show()



    def SaveCellFeatureEvolution (self, feature_names)->list:

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
                                cell_features[feature].append(from_cell.features[feature])

                    # save the other feature values
                    if to_cell.time_point not in cell_features["time_points"] : # take randomly one forward cell (in the case of unidirectional tracking
                                                                                # which is non optimal). once this problem is solved, the registration of
                                                                                # all the features of the new cell must be done
                                                                                #(from t0 : take its mother cell features, to the end)

                        cell_features["time_points"].append(to_cell.time_point)
                        cell_features["uid"].append(to_cell.uid)
                        cell_features["position"].append(to_cell.position)
                        cell_features["pixels"].append(to_cell.pixels)
                        for feature in feature_names:
                            cell_features[feature].append(to_cell.features[feature])
                        if cell_features not in self.feature_save:
                            self.feature_save.append(cell_features)
        return self.feature_save


    def CellLabeling(self,channel,features,uid):

        """
        CellLabeling function allows to identify cells in the original image,
        by labeling them with their cell unique identifier.
        """

        if not os.path.exists("outputs/labeled_segmentation"):
            os.makedirs("outputs/labeled_segmentation")

        for idx,frames_ in enumerate (self.frames[channel]):
            pl_.figure(figsize=(30,30))
            pl_.imshow(frames_.contents.T, cmap="gray")

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
            title= "outputs/labeled_segmentation/frame"+str(idx)+".tif"
            pl_.savefig(title)
            pl_.close()

        frames_.ClearContents()
