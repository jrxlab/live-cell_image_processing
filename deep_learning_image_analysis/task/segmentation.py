#!/usr/bin/python3
#-*- coding: utf-8 -*-

from model import unet_arch as unet
import matplotlib.pyplot as pl_
import numpy as np_
import scipy.ndimage as im_
import skimage.morphology as mp_
import os
from run_parameters import cell_model, nucl_model, optimizer, loss, metrics, contrast, percentile
from task.normalization import ContrastNormalized


def CellSegmentation(frames,channel,file_name) :

    if not os.path.exists("output/"+str(file_name)+"/segmentation/cell"):
        os.makedirs("output/"+str(file_name)+"/segmentation/cell")

    frame_shape=frames[0].shape

    model= unet.get_unet(frame_shape[0],frame_shape[1],channel)
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
    model.load_weights(cell_model)


    if contrast.lower()=="yes" :

        bright_frames = []

        for frame in frames :

            frame = ContrastNormalized(frame, percentile)
            pl_.imshow(frame)
            bright_frames.append(frame)

        bright_frames=np_.asanyarray(bright_frames)
        bright_frames = np_.expand_dims(bright_frames, axis=3)
        prediction = model.predict(bright_frames)

    elif contrast.lower()=="no":

        frames=np_.asanyarray(frames)
        frames = np_.expand_dims(frames, axis=3)
        prediction = model.predict(frames)


    segmentation=[]
    thr =0.9

    for idx,pred in enumerate(prediction):

        pred=(pred.reshape(frame_shape[0],frame_shape[1])>thr).astype(np_.uint8)
        pred= mp_.area_closing(pred)
        #pred=mp_.binary_dilation(pred, mp_.disk(2))
        segmentation.append(pred)
        labeled_sgm = mp_.label(pred, connectivity=1)
    #    pl_.figure()
    #    pl_.imshow(pred,cmap="gray")
        pl_.imsave("output/"+str(file_name)+"/segmentation/cell/frame"+str(idx)+".jpg",labeled_sgm,cmap="gray")

    return segmentation




def NuclSegmentation(frames, channel,file_name):

    """
    NuclSegmentation function
    Segments cell nuclei
    """
    if not os.path.exists("output/"+str(file_name)+"/segmentation/nucl"):
        os.makedirs("output/"+str(file_name)+"/segmentation/nucl")

    frame_shape=frames[0].shape
    model= unet.get_unet(frame_shape[0],frame_shape[1],channel)
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])

    model.load_weights(nucl_model)

#    if contrast.lower()=="yes" :
#
#        bright_frames = []
#
#        for frame in frames :
#
#            frame = ContrastNormalized(frame, percentile)
#
#            bright_frames.append(frame)
#
#        bright_frames=np_.asanyarray(bright_frames)
#        bright_frames = np_.expand_dims(bright_frames, axis=3)
#        prediction = model.predict(bright_frames)

    #elif contrast.lower()=="no":

    frames=np_.asanyarray(frames)
    frames = np_.expand_dims(frames, axis=3)
    prediction = model.predict(frames)

    segmentation_org=[]
    segmentation_dil=[]
    segmentation_ero=[]
    thr =0.9

    for idx,pred in enumerate(prediction):

        pred=(pred.reshape(512,512)>thr).astype(np_.uint8)
        pred_org= mp_.area_closing(pred)
        segmentation_org.append(pred_org)

        pred_dil=mp_.binary_dilation(pred_org, mp_.disk(2))
        segmentation_dil.append(pred_dil)

        pred_ero=mp_.binary_dilation(pred_org, mp_.disk(2))
        segmentation_ero.append(pred_ero)
    #    pl_.figure()
    #    pl_.imshow(pred,cmap="gray")
        labeled_sgm = mp_.label(pred_org, connectivity=1)
        pl_.imsave("output/"+str(file_name)+"/segmentation/nucl/frame"+str(idx)+".jpg",labeled_sgm,cmap="gray")

    return segmentation_org, segmentation_dil, segmentation_ero



def CytoSegmentation(cell_seg,nucl_seg,file_name):


    if len(cell_seg) != len(nucl_seg):
        raise ValueError (f"Number of cell frame {len(cell_seg)} different from Number of cell frame {len(nucl_seg)}")

    if not os.path.exists("output/"+str(file_name)+"/segmentation/cyto"):
        os.makedirs("output/"+str(file_name)+"/segmentation/cyto")

    segmentation=[]


    for idx,cell_mask in enumerate(cell_seg):
        mask= np_.zeros((cell_mask.shape))

        # gérer les -1
        for x in range(0,cell_mask.shape[0]):
            for y in range(0,cell_mask.shape[1]):

                #pix=cell_mask[x,y]-nucl_seg[idx][x,y]
                pix=np_.subtract(cell_mask[x,y], nucl_seg[idx][x,y], dtype= np_.float32)
                if pix == -1 :
                    mask[x,y]=0

                else :
                    mask[x,y]=pix

        mask= mp_.binary_closing(mask, mp_.disk(2))

        labeled_sgm = mp_.label(mask, connectivity=1)

        segmentation.append(mask)

        pl_.imsave("output/"+str(file_name)+"/segmentation/cyto/frame"+str(idx)+".jpg",labeled_sgm,cmap="gray")

    return segmentation









def PlotSegmentations(file: str,segmentations, show_figure: bool = True) -> None:

    """
    PlotSegmentations function
    Plots and saves in the output directory the frame segmentation at each time point
    """

    if not os.path.exists("output_"+str(file)+"/segmentation"):
        os.makedirs("output_"+str(file)+"/segmentation")


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

            plot_axes.matshow(segmentations[i].T)

            # save plot
            pl_.savefig("output_"+str(file)+"/segmentation/frame_"+str(i))



    if show_figure:
        pl_.show()
