
##############################################
#                                            #
#         INITIALIZE INPUT PARAMETERS        #
#                                            #
##############################################

#==============================================================================
# Do you want to train a new segmentation model  ?

new_model= "no"  # "cell_model" - "nucl_model"  - "cell_nucl_model"

cell_model="path/name.h5" # name of a saved model / name of the new model 
                    #(different from the saved model names)
                    
nucl_model="path/name.h5"

#--- if new_model="yes" :

training_data_path=None
seg_data_path=None

gray_rgb= 1  # if gray scale img 1 
             # if rgb 3

#==============================================================================

#Directory path

sequence_path = "C:/Users/ch_95/Desktop/unet_cell_analysis/test"


#    sequence_path = directory+"/"+fname

n_channels= 3 # channel number

# number of frames to treat per channel

from_frame = 0  # first frame to read
to_frame =5     # last frame to read



acq_seq = ["yfp","cherry","___"]   # acquisition_sequence_of_timelapse_images
                                # if channel_name = "___" (three underscores), the channel won't be read
                                #---> ["channel_1_name","channel_2_name",..] 


#============ Contrast Normalization ============#

percentile= [ 20, 80 ]    # interval for intensity streching
                          # exemples : [10,90] for DV images, [2,98] for Operetta images
                          # ---> [min_percentile, max_percentile]

#========= Between channels shift correction =========#

ref_channel = "yfP"  # reference channel : to correct the between channels shift
                     #---> "channel_name" 


#============ Segmentation ============#

segm_protocol ="cell and nucl" # segmentation protocol : choose cellular , cytoplasmic or nuclear
                       # ---> "cell" OR "nucl" OR "cell and nucl"
                       
# if "cell and nucl" : Do you want to get cyto segmentation ?

cyto_seg= "yes"  # "yes" OR "no"

# choose segmentation channels 

cell_channel ="yfp"    # Cell segmentation channel : channel to use for cell segmentation
                       #---> "cell_channel_name"  OR None

nucl_channel="cherry"   # Nucleus segmentation channel : channel to use for nuclear segmentation
                   #---> "nucl_channel_name"  OR None

#============ Signal extraction ============#
                   

# Do you want to use the same channel to extract signal? "yes" / "no"
                   
same_channel="yes"  
                   
# if same_channel="yes" : which channel to use ?

signal_channel="yfp"
                   
# if same_channel="no" : you can choose many channels per type 

cell_sig=["yfp"] 
nucl_sig=["yfp"]  
cyto_sig=["yfp"]    # None if no cyto segmentation              
         
         
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   
                   

uid= "root_cell_uid"   # cell unique identifier for labeling cells in the original image
                       # "root_cell_uid" : the same cell identifier over the frames
                       # "cell_uid": per frame cell identifier
                       # ---> [ "root_cell_uid" ] OR [ "cell_uid" ]






#========= Feature computation =========#

fret_ratio=["yfp","yfp"] # channels to use to calculate the FRET ratio
                         # ---> [["numerator", "denominator"]]

edge_channel="yfp"       # channel to use for edginess calculation
                         # ---> ["channel_name"]

#### if "RFP" channel exists ###

channel_treat_prot = "cyto" # channel_treatment_protocol : choose the protocol of the "RFP" channel treatment
                            # "cyto" : get the global cell intencities - "MOMPloc" : detetct MOMP event
                            # ---> [ "cyto" ] OR ["MOMPloc"]

# if "MOMPloc" is choosed

momp_channel="yfp"  # channel to use to calculate the MOMP location in addition to RFP channel
                    # ---> ["channel_name"]

#============ Tracking ================#

# which segmentation to use "cyto" or "nucl" (if cyto & nucl) !!!!!!!!!!!!!!!!

bidirectional= True  # tracking method, could be bidirectional or not
                     #---> [ True ] OR [ False ]
