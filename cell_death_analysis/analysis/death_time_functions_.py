import numpy as np_
import scipy as sc_



def FilterSmooth(feature:np_.array,idx_last:int,windowSize):
    """
    FilterSmooth function
    This function filters image by a forward-backward filter or by a linear convolution
    """
    
    if idx_last > ( 3*windowSize):
        #forward-backward filter
        feature[0:idx_last]= sc_.signal.filtfilt(np_.ones((windowSize))/windowSize,1,feature[0:idx_last])         
    else:
        #discrete linear convolution
        feature[0:idx_last]= np_.convolve(feature[0:idx_last].T,windowSize,mode='same') 
        
        
    return feature[0:idx_last]
    


def EvalTrajNoise(feature,smoothed_feature,noise_idx,windowSize):
    
    """
    EvalTrajNoise function
    Calculates the noise of the input feature
    """
    
    # standard deviation 
    noise=np_.asanyarray( np_.nanstd(feature[5:min(windowSize+5,noise_idx)] - 
                                             smoothed_feature[5:min(windowSize+5 ,noise_idx)]))
    
    noise[ np_.isnan(noise)==True]=100 #replace "nan" by 100
        
    noise[noise>100]=100 #replace all values > 100 by 100
    
    return noise



def DerivateScore(smoothed_edge, idx_last,windowSize,edge_cutoff,margin_delta,MOMP_cutoff):
    
    """
    DerivateScore function
    Calculates the derivate based on the cell edge values 
    """
    
    array=np_.array([0]) # inorder not to have length error:beacause "diff" function deletes the 1st position
    delta= np_.append(array,np_.diff(smoothed_edge-np_.nanmean(smoothed_edge[0:windowSize])/edge_cutoff)) #derivate
  
    delta[delta < -0.0015]= -0.0015
    
    p_delta=np_.zeros((1,int(min(idx_last,MOMP_cutoff-np_.floor(windowSize/2)))))
    
    
    for idx in range (1,len(p_delta[0])):
        
        if delta [idx]> margin_delta: #when apoptosis
            
            p_delta[0,idx]= min(1,max(0,p_delta[0,idx-1])+.05+min(.015,(delta[idx]-
                   margin_delta)/margin_delta))
        elif delta[idx] < -margin_delta:
            p_delta[0,idx]=max(-0.03,min(0,p_delta[0,idx-1])-0.003)
        else:
            p_delta[0,idx]=0 
            
    
    return delta, p_delta



def CumulativeNormalDistribution (feature,mean,std:None):
    
    """
    CumulativeNormalDistribution function
    Calculates the inverse cumulative distribution of the input feature
    """
    
    if std == None or np_.isnan(std) or std==0:
        if np_.isnan(mean):
            feature=feature
        else:
            feature= feature -mean
        
    elif std != None:
        feature=(feature-mean)/std 
    
    # icdf : Inverse cumulative distribution function     
    norm_icdf={}
    
    if len(norm_icdf)==0:
        step = 2e-5
        step2 = step/1e3
        
        # probability array : form 0 to 1 
        norm_icdf["proba"]=np_.array([0 ,step/2])
        norm_icdf["proba"]=np_.append(norm_icdf["proba"],np_.arange(step,(1-step)+step,step))
        norm_icdf["proba"]=np_.append(norm_icdf["proba"],1-(step/2))
        norm_icdf["proba"]=np_.append(norm_icdf["proba"],1)
        norm_icdf["proba"]=np_.append(norm_icdf["proba"],0.001+ np_.arange((-step-step2/2),
                (step+step2/2)+step2,step2))
        #print(f"{len(norm_cdf['y'])}")
        norm_icdf["proba"]=np_.append(norm_icdf["proba"],0.01+np_.arange(-step-(step2/2),
                (step+step2/2)+step2,step2))
        norm_icdf["proba"]=np_.append(norm_icdf["proba"],0.05+np_.arange((-step-step2/2),
                (step+step2/2)+step2,step2))
        #print(f"{len(norm_cdf['y'])}")
        norm_icdf["proba"]=np_.append(norm_icdf["proba"],0.95+ np_.arange((-step-step2/2),
                (step+step2/2)+step2,step2))
        norm_icdf["proba"]=np_.append(norm_icdf["proba"],0.99+np_.arange((-step-step2/2),
                (step+step2/2)+step2,step2))
        norm_icdf["proba"]=np_.append(norm_icdf["proba"],0.999+np_.arange((-step-step2/2),
                (step+step2/2)+step2,step2))
    
    norm_icdf["proba"]=np_.unique(norm_icdf["proba"]) 
   
    #Inverse cumulative distribution 
    norm_icdf["icdf"]=sc_.stats.norm.ppf(norm_icdf["proba"],0,1) # (0,1) for normal distribution
   
    if not np_.isnan( feature ):
        
        idx=[idx for idx, val in enumerate(norm_icdf["icdf"])  
                                        if val > feature]
        if len(idx)==0:
            cdf_prob=1
            return cdf_prob

        idx=min(idx)
        cdf_prob= norm_icdf["proba"][idx-1]+ np_.diff(np_.array([norm_icdf["proba"][idx-1],
                            norm_icdf["proba"][idx]]))* (feature-norm_icdf["icdf"][idx-1])/np_.diff(np_.array(
                [norm_icdf["icdf"][idx-1],norm_icdf["icdf"][idx]])) 
        
        if np_.isnan(cdf_prob): # to avoide "nan" when dividing by "inf"
            
            cdf_prob=0
    else:
        cdf_prob=0
    
    return cdf_prob


def CutoffDistProb(feature, sliding_cutoff,edge_cutoff,noise,pre_w,pre_2w, post_w, death_w):
    """
    CutoffDistProb function
    Calculates edge based probabilities 
    """
    prob=np_.empty((2,1))
    prob[0,0] = max(0, 
               min(1,sum(feature[post_w]>sliding_cutoff+noise)/max(len(post_w),5)
               - .5*sum(feature[pre_w]>sliding_cutoff+0.2*edge_cutoff+noise)/max(len(pre_w),5)
               - .25*sum(feature[pre_w]>(sliding_cutoff+1.5*edge_cutoff+noise))/max(len(pre_w),5)
               - sum(feature[death_w]<sliding_cutoff)/max(len(death_w),5)))

                                 
   
    prob[1,0]=max(0,(CumulativeNormalDistribution(min(feature[death_w]),np_.mean(feature[pre_2w-2]),
                                       noise+np_.std(pre_2w-2))-.9)/.1)

    
    return prob



def EvaluateCellDeath (prob_MOMP,prob2_MOMP,delta_D_MOMP,MOMP_cutoff,p_cutoff_high,p_cutoff_low):
    
    """
    EvaluateCellDeth function 
    This function returns a cell classification:
    inf        -> survivor
    frame num  -> momp occures at "frame num"
    -2         -> unclassified
    
    """
       
    if any (prob2_MOMP[0:MOMP_cutoff,0]>p_cutoff_high):
       
        time=max(idx for idx, val in enumerate(prob2_MOMP[0:MOMP_cutoff])  
                                    if val > p_cutoff_high)
        momp= min(idx for idx, val in enumerate(prob2_MOMP[0:time+1])  
                                    if val > p_cutoff_high)
        
        if time != momp:
            momp_range=np_.arange(momp,time+1)
            
            momp_time=max(idx for idx, val in enumerate(prob2_MOMP[momp_range])  
                                        if val > max(val)*.95 and 
                                            val >p_cutoff_high
                                        )      
            momp_time+= momp_range[0]-1-delta_D_MOMP # momp frame num         
        else:
            momp_time= time-1-delta_D_MOMP # momp frame num   
           
    elif (np_.sum(prob_MOMP[:,0:MOMP_cutoff])< p_cutoff_low):
        
        momp_time= np_.inf # survived
        
    else:
        momp_time= -2 #unclassified

    return momp_time



def PrintResult(edge): 
    
    """
    PrintResult function
    Prints the cell fate (classification) result
    """      
    survivor=len(edge[edge==np_.inf])
    dead =len(edge[edge<np_.inf])-len(edge[edge<0])
    too_short=len(edge[edge==-1])
    unclassified =len(edge[edge==-2])
    print(f"\n-------- Cell classification --------\nsurvivors : {survivor} \n"+
            f"dead : {dead}\n"+
            f"too short : {too_short}\n"+
            f"unclassified : {unclassified} ")