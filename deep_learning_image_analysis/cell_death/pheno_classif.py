#!/usr/bin/python3
#-*- coding: utf-8 -*-

import pathlib
import numpy as np_
import pylab as pl_
import scipy as sp_
import pandas as pd_
from sklearn.metrics import roc_auc_score , confusion_matrix, plot_confusion_matrix
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import linear_model
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt



def GetCellDeathMatrix(folders: list):
    
    feat_matrix=np_.empty((10000,12))*np_.NaN
    class_matrix=np_.empty((10000,1))*np_.NaN
    row_prev=0
    
    print("\nLoad :")
    for folder_name in folders :
        
        gt_path = pathlib.Path("C:/Users/ch_95/Desktop/cell_death/"+folder_name+"/ground_truth")
        
        
        print(f"\n{folder_name}")
        
        for frame_folder in gt_path.glob("frame_*"):
           
            feat_file = str(frame_folder)+"\cells.csv"
            class_file= str(frame_folder)+"\classes.csv"
             
            features= pd_.read_csv(feat_file, header=None)
            features=features.iloc[:,:].values
            
            class_= pd_.read_csv(class_file,header=None)
            
            class_[class_=="R"]=0
            class_[class_=="S"]=1
            class_[class_=="D"]=2
            
            class_=class_.iloc[:,:].values
            
            row= np_.shape(features)[0]
            col= np_.shape(features)[1]
            
            feat_matrix[row_prev:row+row_prev,:col]= features[:,:]
            class_matrix[row_prev:row+row_prev,0]= class_[:,0]
            
            row_prev+=row
    
    mask_class = np_.all(np_.isnan(class_matrix) , axis=1)           
    feat_matrix=feat_matrix[~mask_class]
    class_matrix= class_matrix[~mask_class]
    
    mask_feat = np_.all(np_.isnan(feat_matrix) , axis=1) 
    feat_matrix=feat_matrix[~mask_feat]
    class_matrix= class_matrix[~mask_feat]
    
    
    R= len(class_matrix[class_matrix==0])
    print(f"\nR : {R}")
    
    D= len(class_matrix[class_matrix==2])
    print(f"D : {D}")
    
    S= len(class_matrix[class_matrix==1])
    print(f"S : {S}")
    
    
    
    return feat_matrix.astype("float32"), class_matrix.astype("float32")




def TrainRF (train_feat, train_class, test_feat, test_class, model_name):
    
    print("\nTrain Random Forest model ...")
    
    clf = RandomForestClassifier(n_estimators=1500 , max_depth=20,random_state=5,class_weight="balanced_subsample")

#    cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=3)
#    scores=cross_val_score(clf, train_feat, train_class, cv=cv,scoring='accuracy')
#    score=np_.mean(scores)
    
    clf.fit(train_feat,train_class.ravel())
    pickle.dump(clf, open(model_name, 'wb'))
    
#    print(f"Training score : {score}")
    

    print("\nPrediction ...")
    
    plot_confusion_matrix(clf,test_feat,test_class,display_labels=["resistant","sensitive","division"], cmap="Blues")
    plt.show()
    
    
    print("\nFeature importance ...")
    
    
    feature_importance= clf.feature_importances_

    fig, ax = plt.subplots()
    
    features = ('area', 'edge','shift', 'bbox_area', 'convex_area','eccentricity','equivalent_diameter','maj_axis_len','min_axis_len',
                'perimeter','mean_intencity','median_intencity')
    y_pos = np_.arange(len(features))
    
    
    ax.barh(y_pos, feature_importance, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Importance')
    ax.set_title('Feature importance')
    
    plt.show()
    
    
    
    
###### MAIN #####

# TRAIN / TEST MATRICES PREPARATION
    
train_folders=["treat01_47_R3D","treat01_46_R3D","treat01_59_R3D","treat01_50_R3D","treat01_48_R3D","treat01_49_R3D"]
test_folders=["treat01_12_R3D","treat01_60_R3D"]

print("\n*** Train ***")
feat_train, class_train=GetCellDeathMatrix(train_folders)

print("\n*** Test ***")
feat_test, class_test=GetCellDeathMatrix(test_folders)


# TARIN RF MODEL 

model_name = 'RF1_crossval_weighted_model.sav' 

TrainRF (feat_train,class_train, feat_test, class_test, model_name)


