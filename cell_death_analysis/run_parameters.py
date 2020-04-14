
##############################################
#                                            #
#         INITIALIZE INPUT PARAMETERS        #
#                                            #         
##############################################


sequence_path = "C:/Users/ch_95/Desktop/stage_M2/code/image_new/pretreat01_40_R3D.tif"

n_channels= 3 # channel number

# number of frames to treat per channel

from_frame = 0  # first frame to read
to_frame =9     # last frame to read



acq_seq = ["cfp","yfp","___"]   # acquisition_sequence_of_timelapse_images 
                                # if channel_name = "___" (three underscores), the channel won't be read                                
                                #---> [ ["channel_1_name","channel_2_name",..] ]


#============ Contrast Normalization ============#
                              
percentile= [ 10, 90 ]    # interval for intensity streching 
                          # exemples : [10,90] for DV images, [2,98] for Operetta images
                          # ---> [min_percentile, max_percentile]

#========= Correction of the chromatic aberration =========#
                                
ref_channel = "cfP"  # reference channel : to correct the chromatic aberration 
                     #---> [ "channel_name" ]


#============ Segmentation ============#
                     
segm_protocol = "cyto" # segmentation protocol : choose cytoplasmic or nuclear
                       # ---> [ "cyto"] OR ["nucl"]
                       
segm_channel = "yfp"   # segmentation channel : the channel to use for segmentation 
                       #---> [ "channel_name" ]

show_sg_steps= True    # show the first n_frames segmentation steps 
                       #---> [ True ] OR [ False ]
                     
n_frames= 9            # number of frames to show the segmentation steps if "show_sg_steps= True "
                       # ---> [ num ] : from "first frame to read" to "last frame to read"  !

                     
uid= "cell_uid"        # cell unique identifier for labeling cells in the original image
                       # "root_cell_uid" : the same cell identifier over the frames
                       # "cell_uid": per frame cell identifier 
                       # ---> [ "root_cell_uid" ] OR [ "cell_uid" ]
                     
#========= Feature computation =========#
                       
fret_ratio=["cfp","yfp"] # channels to use to calculate the FRET ratio
                         # ---> [["numerator", "denominator"]]
                         
edge_channel="yfp"       # channel to use for edginess calculation      
                         # ---> ["channel_name"]
                     
#### if "RFP" channel exists ###
                         
channel_treat_prot = "cyto" # channel_treatment_protocol : choose the protocol of the "RFP" channel treatment
                            # "cyto" : get the global cell intencities - "MOMPloc" : detetct MOMP event 
                            # ---> [ "cyto" ] OR ["MOMPloc"]

# if "MOMPloc" is choosed

momp_channel="cfp"  # channel to use to calculate the MOMP location in addition to RFP channel
                    # ---> ["channel_name"]

#============ Tracking ================#
                            
bidirectional= True  # tracking method, could be bidirectional or not
                     #---> [ True ] OR [ False ]



             

