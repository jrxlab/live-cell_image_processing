import numpy as np_
from analysis import death_time_functions_ as df_
from analysis.death_time import CellDeath_t
from run_parameters import to_frame



class process_traj:
    
    def __init__(self)->None:
        
        self.max_time = to_frame*3  #number of minutes for full tracking
        self.min_time = self.max_time/2  # for a full tracking except for cell death
        self.time_afterMOMP = self.max_time # Time after MOMP for trajectories
        
        
    
    @classmethod 
    
    def EvalTrack(cls,death_param):
        
        """
        EvalTrack function
        Corrects the tracks
        Error type :
        The determination of the error types depends on both cell classification and track lost
        0  -> good tracking
        1  -> bad tracking
        2  -> tracking too short
        12 -> tracked to the minimun time
        3  -> undetermined fate
        4  -> momp recorded too early
        5  -> unexpected case    
        """
        
        instance=cls()
        
        
        ##
        
        instance.full_time=np_.arange(0,death_param.Ntimepoints)*death_param.Timestep
        instance.smoothing= max(death_param.windowSize-2,5)
        instance.windowsize= round(death_param.windowSize+10/death_param.Timestep) 
        instance.min_len=round(50/death_param.Timestep) #minimum number of frame for considering a good tracking
        
        
        cutoff=min(time for time, val in enumerate(instance.full_time)  
                                    if val >= instance.max_time ) # survival cutoff: min cutoff to consider surviving cells
        
        cutoff2= cutoff + death_param.windowSize+2
        instance.time= instance.full_time[0:cutoff] 
        T2=instance.full_time[0:cutoff2] #slightly longer T for smoothing
       
        # application of the survival cutoff
        #surviving cells
        with np_.errstate(invalid='ignore'):
            death_param.cell_fate[:,1][death_param.cell_fate[:,1]>=cutoff]=np_.inf # replace all fates >= cutoff by "inf" 
            death_param.track_lost[death_param.track_lost[:,1] # replace all track_lost >= cutoff by "nan"
                                            >=cutoff]= np_.NaN                  # so the track is not lost
                                       
        
        #remove the bad tracks 
        instance.good_track= np_.zeros((death_param.n_cell,1),dtype=bool) # array of good tracked cells
        instance.error_type=np_.zeros((death_param.n_cell,1)) # array of error type of bad tracked cells
        goodtrack=instance.good_track
        error_type=instance.error_type
        
        for traj in range (0,death_param.n_cell):
            
            if death_param.cell_fate[traj,1]==-1: # bad tracking
                error_type[traj,0]=1
            elif death_param.cell_fate[traj,1]==-2: # undetermined fate
                error_type[traj,0]=3
            elif np_.isinf(death_param.cell_fate[traj,1]): 
                
                # survived cells (need to be tracked until the end)
                goodtrack[traj,0]=np_.isnan(death_param.track_lost[traj,1]) # track_lost = Nan => good track
               
                if not goodtrack[traj,0]:
                    
                    if T2[int(death_param.track_lost[traj,1])] > instance.min_time:
                        #print(" tracked to the min time ")
                        error_type[traj,0]=12 # tracked to the minimum time
                    else:
                        #print("tracking too short")
                        error_type[traj,0]=2 # tracking too short
            elif death_param.cell_fate[traj,1] >0 and death_param.cell_fate[traj,1] < np_.inf and (death_param.cell_fate[traj,1] >= instance.min_time
                                          or np_.isnan(death_param.track_lost[traj,1])):
                #momp recorded : need to be trackes until momp at least
                if death_param.cell_fate[traj,1] <= death_param.track_lost[traj,1] or np_.isnan(death_param.track_lost
                                            [traj,1]): 
                    goodtrack[traj,0]= True
                else:
                    error_type[traj,0]=5
                    Warning("Unexpected case")
                    
            else: # momp recorded too early
                assert death_param.track_lost[traj,1] < instance.min_len, \
                       f"Track lost on frame: {death_param.MOMPtime_track_lost[traj,1]} >\
                       minimum number of frame for considering a good tracking: {instance.min_len}"
                error_type[traj,0]=4
        
         
        assert len(goodtrack[goodtrack==True])==len(error_type[error_type==0]), \
                "Error in good track identification"
        
        good_track=len(error_type[error_type==0])
        bad_track=len(error_type[error_type==1])
        too_short=len(error_type[error_type==2])
        track_min =len(error_type[error_type==12])
        undetermined_fate =len(error_type[error_type==3])
        early_momp =len(error_type[error_type==4])
        unexpected_case =len(error_type[error_type==5])
        print(f"\n-------- Trajectory Processing  --------\nGood tarcking : {good_track} \n"+
                f"Bad tracking : {bad_track} \n"+
                f"Tracking too short : {too_short} < Less then {instance.min_len} frames > \n"+
                f"Tracked to the min time : {track_min} \n"+
                f"Undetermined fate : {undetermined_fate} < Tracking lost between {instance.min_time}mn\
                and {instance.max_time}mn >\n"+
                f"Early MOMP : {early_momp}\n"+
                f"Unexpected case : {unexpected_case}\n")  
        
        print("Cell Classification After Trajectory Processing")
        df_.PrintResult(death_param.cell_fate[:,1])
        return instance               
                
