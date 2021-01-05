# Live-cell image processing
**Author :** Asma Chalabi, code developed in the [**Roux lab**](https://github.com/jrxlab) 


**Note 1:** Both pipelines are developed under Python 3.7 using Spyder software from Anaconda platform.

**Note 2:** The *deep_learning_image_analysis* pipeline is still under development.


Both pipelines run an analysis of live-cell microscopy stacks, the stack is split into different channels, then each of them is treated separately. 

**1. cell_death_analysis** 

This pipeline is based on "classic" image processing approaches. It uses Gaussian smoothing for noise reduction.The segmentation is performed by a binary thresholding followed by a watershed segmentation.  

Only cell segmentation is performed with this pipeline, the different cell intensities and features are extracted after the cell tracking, then the cell death detection is performed based on the edge metric parameter. 



**2. deep_learning_image_analysis**

This pipeline is based on machine learning approaches. By integrating a deep learning segmentation model (U-Net), the pipeline can manage 3 different segmentations : cellular, nuclear and cytoplasmic at the same time. The tracking is then performed,  the features of each structure and the signal intensities are extracted, and the statistics are then computed on the extracted signal (mean, median, ratio).  

The cell phenotypes (sensitive, resistant, division) are then classified by using a Random Forest classifier trained on the cell features. (This part is still in under development)