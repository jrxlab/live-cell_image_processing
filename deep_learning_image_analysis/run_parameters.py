#!/usr/bin/python3
#-*- coding: utf-8 -*-


           ##############################################
          #                                            #
         #         INITIALIZE INPUT PARAMETERS        #
        #                                            #
       ##############################################





#============= Do you want to train a new segmentation model  ? ===============

new_model= "no" ### if YES :

               # Choose ---> "cell_model" OR "nucl_model"  OR "cell_nucl_model"

               ### if NO :

               # Choose ---> "no"



#--- if train new model : 

training_masks_path=None # Path to the binary masks file
                         # if new_model="no" ---> None
                         
training_img_path=None   # Path to the training images file
                         # if new_model="no" ---> None



              
                
#====== PATH TO THE SEGMENTATION MODEL =======#
             
### if train new model :
                
# enter the path where to save the new model ----> " path/model_name.h5"
# NB: BE CAREFUL NOT TO USE AN ALREADY EXISTING NAME !

### if use a pre-trained model : 

# enter the path to the pre-trained model file


cell_model="C:/Users/ch_95/Desktop/live-cell_image_processing/deep_learning_image_analysis/trained_models/hela_cell_seg_model.h5" 
                    
nucl_model="C:/Users/ch_95/Desktop/live-cell_image_processing/deep_learning_image_analysis/trained_models/hela_nucl_seg_model.h5"




### Model parameters :

validation_split=0.2

batch_size=10

epochs=30 

# THE PARAMETERS BELOW MUST BE ENTRED WHETHER YOU TRAIN OR NOT A NEW MODEL
# IF A PRE-TRAINED MODEL IS USED, THESE PARAMETERS CORRESPOND TO THOSE
# USED TO TRAIN THIS MODEL.

        
                 
gray_rgb= 1  # if gray scale images ---> 1 
             # if rgb images ---> 3
                
optimizer="adam"

loss="binary_crossentropy"

metrics="accuracy"




#=====================     ANALYSE NEW DATA     ===============================


# NB: IMAGES TO ANALYSE MUST HAVE THE SAME SHAPE OF THE TRAINING IMAGES


# path to the DIRECTORY to analyse 

sequence_path = "C:/Users/ch_95/Desktop/live-cell_image_processing/deep_learning_image_analysis/data" 


n_channels= 3 # the REAL channel number

# number of frames to treat per channel

from_frame = 0  # first frame to read
to_frame =200   # last frame to read



acq_seq = ["yfp","cherry","___"]#["yfp","cherry","___"]   # acquisition_sequence_of_timelapse_images
                                   # if channel_name = "___" (three underscores), the channel won't be analysed
                                   # ---> ["channel_1_name","channel_2_name",..] 


#============ Contrast Normalization ============#  ::::::: PROBLEME ! ::::::::
                                   
# Do you want to enhace contrast ? 
                                   
contrast= "no"     # "yes" OR "no"

percentile= [ 10, 90 ]    # interval for intensity streching
                          # exemples : [10,90] for DV images, [2,98] for Operetta images
                          # ---> [min_percentile, max_percentile]
                          
                          

#========= Between channels shift correction =========#

ref_channel = "yfP"  # reference channel : to correct the between channels shift
                     #---> "channel_name" 



#============ Segmentation ============#

segm_protocol = "cell and nucl" # segmentation protocol : cellular, nuclear or both
                      # Choose ---> "cell" OR "nucl" OR "cell and nucl"
                       
# if segm_protocol="cell and nucl" : Do you want to perform cyto segmentation ?

cyto_seg= "yes"  # "yes" OR "no"

# choose segmentation channels 

cell_channel ="yfp"    # Cell segmentation channel : channel to use for cell segmentation
                       # ---> "cell_channel_name"  OR None

nucl_channel="cherry"  # Nucleus segmentation channel : channel to use for nuclear segmentation
                       # ---> "nucl_channel_name"  OR None



#============ Tracking ================#

# which segmentation to use "cell" or "nucl"  
                       
track_channel= "nucl"
bidirectional= False  # tracking method, could be bidirectional or not
                     #--->  True  OR  False 

                       

#============ Signal an features extraction ============#
                   

                     
# Do you want to use the same channel to extract signal? 
                     
## TO USE ONLY IF "2 OR 3" TYPES OF SEGMENTATION ARE DONE AND THE SIGNAL TO EXTRACT IS "THE SAME" FOR EACH TYPE
                   
same_channels=["yfp"] # ---> ["channel1", "channel2",..] OR None


## IF "1, 2 OR 3" TYPES OF SEGMENTATION ARE DONE AND THE SIGNAL TO EXTRACT IS "NOT" THE SAME FOR EACH TYPE:                                 

# Put:  same_channel=None 

# You can choose many channels per segmentation type 
# ---> ["channel1", "channel2",..] OR None

cell_sig= None
nucl_sig=None 
cyto_sig=None  # if no cyto segmentation ---> None   
       
         
# If different channels are used to extract the signal :

# Which channel to use to extract features ?

feature_channel="yfp"  # ---> "channel" OR None

   
#============ Statistic operations ============#                 
                   
# Which operatons to perform on the interested signal ?

# Choose ---> True OR False

mean= True 
median= True
ratio=True     
                   
# if ratio = True :

numerator="yfp"
num_struct= "cyto"

denominator="yfp"                 
denom_struct="nucl"
                













                   

#========= Cell labeling =========#
                   
uid= "root_cell_uid"   # cell unique identifier for labeling cells in the original image
                       # "root_cell_uid" : the same cell identifier over the frames
                       # "cell_uid": per frame cell identifier
                       # ---> [ "root_cell_uid" ] OR [ "cell_uid" ]




