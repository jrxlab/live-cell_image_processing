import numpy as np_
from analysis import death_time_functions_ as df_
from run_parameters import to_frame
import warnings


class CellDeath_t :
    
    
    def __init__(self) -> None:
         
        # Initialize parameters     
        self.Timestep= 3
        self.Ntimepoints= to_frame *3 #to_frame*3 : because 3mn between two time points
        self.windowSize = round(45/self.Timestep) # windows size for filtering
        self.division_time = round(250/self.Timestep) # assumed division time
        self.edge_cutoff = 1e-02 # cutoff for the edge criterion
        self.margin_delta = .001 # hyteresis for the scoring of the derivate
        self.p_cutoff_high = 1.1 #lower cutoff for the score to be classified as MOMP
        self.p_cutoff_low = .5   # upper cutoff for the score to be classified as survival
        self.MOMP_cutoff = self.Ntimepoints-self.windowSize
        self.Delta_Death_MOMP = 4   # number of frame for MOMP to occur before cell change shape
    

   
    @classmethod 
         
    def EvalCellDeathTime(cls, features: list):
        
        """
        depending on the calculated probability, cells are classified into:
          inf         ->  surviving cells
          frame num   ->  MOMP at frame num 
          -1          ->  track too short
          -2          ->  unclassified 
        """
      
        instance= cls()
        
        instance.n_cell=len(features)
        
        with np_.errstate(invalid='ignore'):
            instance.cell_fate = np_.empty((instance.n_cell,2))*np_.NaN # cell fate
            instance.track_lost = np_.empty((instance.n_cell,2))*np_.NaN # last uid and the frame num of the track lost
            instance.smoothed_Edge = np_.empty((instance.MOMP_cutoff,instance.n_cell))*np_.NaN # smoothes edge
            instance.smoothed_Area = np_.empty((instance.MOMP_cutoff,instance.n_cell))*np_.NaN # smoothed area
            instance.scoring = np_.empty((instance.MOMP_cutoff,instance.n_cell))*np_.NaN # score
            instance.proba_p=[] #probability
        
           
        for i in range(0,instance.n_cell):
            edge=np_.asanyarray(features[i]["edge"]) # cell edge
            area=np_.asanyarray(features[i]["area"]) # cell area

            
            # get the last index
            idx_last=max(idx for idx, val in enumerate(edge)  
                                    if not np_.isnan(val))
            
            
            if not idx_last or  (idx_last and idx_last < instance.windowSize) :
                
                instance.cell_fate[i,0]= features[i]["uid"][0] # get the root cell uid
                instance.cell_fate[i,1] = -1
            
            elif idx_last:
                 instance.track_lost[i,0] = features[i]["uid"][0] # get the root cell uid
                 instance.track_lost[i,1] = idx_last # get the last frame num
                 
            else:
                 ValueError ("__Unexpected Case__")  
                
            idx_last=int(min(idx_last, instance.MOMP_cutoff+np_.floor(instance.windowSize/2)))
            
            # smoothing
            smoothed_edge=df_.FilterSmooth(edge,idx_last,instance.windowSize)
            
                      
            smoothed_area=df_.FilterSmooth(area,idx_last,instance.windowSize)
            
            
            #------ Edge
            instance.smoothed_Edge[0:min(idx_last,instance.MOMP_cutoff),i]=smoothed_edge[0:min(idx_last+1,
                      instance.MOMP_cutoff)]
           
            
            #------ Area
            instance.smoothed_Area[0:min(idx_last,instance.MOMP_cutoff),i]=smoothed_area[0:min(idx_last+1,
                      instance.MOMP_cutoff)]
   
            
            
            # sliding edge cutoff
            sliding_edge_cutoff=instance.edge_cutoff+ np_.nanmedian(smoothed_edge[4:(instance.windowSize + 5)])  
            
            
            
            p=np_.zeros((4,len(edge)))
            p2=np_.zeros((len(edge),1))
            
            
            # find noise first index
            if not np_.isnan(sliding_edge_cutoff):
                
                idx=[idx for idx, val in enumerate(edge[4:])
                            if val > sliding_edge_cutoff]  
                #print(idx)
                if len(idx)!=0:
                    noise_idx=min(idx) +3
                    #print(f"noise idx : {noise_idx}")
                else:
                    noise_idx=None
                   
            
                if  noise_idx!=None:
                    
                # get noise
                    edge_noise= df_.EvalTrajNoise(edge, smoothed_edge,noise_idx,instance.windowSize)
                    
                    area_noise= df_.EvalTrajNoise(area, smoothed_area,noise_idx,instance.windowSize)
                    
                    
                    #score based on the derivate of the edge
                    delta, p_delta=df_.DerivateScore(smoothed_edge, idx_last,instance.windowSize,
                                                     instance.edge_cutoff,instance.margin_delta,instance.MOMP_cutoff)
                   
                    p[3,0:len(p_delta[0])]=.4 * p_delta 
                    
                    for idx in range(instance.windowSize,idx_last):
                        pre_2w=np_.arange(max(instance.windowSize-2,idx-2*instance.windowSize),(idx))
                        pre_w=np_.arange(max(instance.windowSize-2,idx-instance.windowSize+1),(idx))
                        post_w=np_.arange(idx, min(np_.array([max(len(smoothed_edge)-5,
                                                                      idx+instance.windowSize),idx+2*instance.windowSize,
                                                                      len(smoothed_edge)])))
                        death_w=np_.arange(idx,min(np_.array([idx+instance.division_time,
                                                                   max(idx+5,len(smoothed_edge)-5),
                                                                   len(smoothed_edge)])))
                
        
                        if sliding_edge_cutoff > np_.nanmean(smoothed_edge[pre_w])+ edge_noise/2:
                            sliding_edge_cutoff=instance.edge_cutoff+np_.nanmean(smoothed_edge[pre_2w])
            
                       # criteria base on edge : cutoff and ranksum test?
                        prob=df_.CutoffDistProb(smoothed_edge,sliding_edge_cutoff,instance.edge_cutoff,
                                            edge_noise, pre_w,pre_2w,post_w,death_w
                                            )
                        
                        # criteria based on edge cutoff
                        
                        p[0,idx]=.8*prob[0,0]
                        p[1,idx]=.4*prob[1,0]
                        
                        # criteria based on area
                
                        p[2,idx]=.3* max(0,(.05 -df_.CumulativeNormalDistribution(max(smoothed_area[death_w]), 
                                            np_.mean(smoothed_area[pre_2w-1]), area_noise+np_.std(smoothed_area[pre_2w-1]))
                        )/.05)-.2* max(0,(df_.CumulativeNormalDistribution(max(smoothed_area[death_w]), 
                                       np_.mean(smoothed_area[idx:max (post_w-1)]), area_noise+np_.std(
                                               smoothed_area[idx:max(post_w-1)])) -.8)/.2)
                       
                        #contibution for end of tracking
                        
                        p[3,idx]=p[3,idx]* (1+ .5*max(0, min(1,(instance.windowSize
                         -(idx_last-i-4))/(instance.windowSize)))*(idx_last<instance.MOMP_cutoff))
                        
   
                        if smoothed_edge[idx]>(sliding_edge_cutoff - edge_noise):
                            p2[idx,0]=np_.sum(p[:,idx])
                            
                    
            
            instance.proba_p.append(p)
            if instance.cell_fate[i,1] != -1 : # to save the -1 
                instance.cell_fate[i,0]= features[i]["uid"][0] # get the root cell uid
                instance.cell_fate[i,1]= df_.EvaluateCellDeath(p, p2, instance.Delta_Death_MOMP, 
                                      instance.MOMP_cutoff, instance.p_cutoff_high, instance.p_cutoff_low)
            
            instance.scoring[0:min(len(p2),instance.MOMP_cutoff),i:i+1]= p2[0:min(len(p2),instance.MOMP_cutoff)]
            
            
        df_.PrintResult(instance.cell_fate[:,1]) # print the evaluation results
        
        warnings.filterwarnings("ignore")  
        
        return instance
    



