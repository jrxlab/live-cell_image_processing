# Cell-based pharmacodynamics toolset using deep learning image analysis 
**Author :** Asma Chalabi, code developed in the [**Roux lab**](https://github.com/jrxlab) 


**Note 1:** This pipeline is still under development.

**Note 2:** This pipeline is being developed under Python 3.7 using Spyder software from Anaconda platform.


The pipeline runs an analysis of live-cell microscopy stacks, it splits the stack into different channels, then treat each of them separately. 

It performs 3 different segmentations : cellular, nuclear and cytoplasmic using U-Net deep learning architecture model, cell tracking and cell signal and different cell features extraction. For that, it gives
the choice to use a trained model (transfer learning) or to train a new model, this flexibility will make it
usable for different cell types.

### The code is split into multiple scripts contained in different directories:

**- run_parameters.py :**

  File to initialize the input parameters.

**- run_pipeline.py :**

  This is the file to execute to start the analysis.

**- model directory :**

  - unet_arch.py : Contains the U-Net architecture model.
  - train_new_model.py : New model training script.

**- trained_models directory :**

  Contains the segmentation trained models weights :
   - hela_cell_seg_model.h5 : HeLa cell segmentation trained model.
   - hela_nucl_seg_model.h5 : HeLa nuclear segmentation trained model.

**- task directory :**

  Contains different tasks scripts that the pipeline offers,
  like segmentation and feature extraction.

**- type directory :**

  Contains different scripts to save features of the different types that we can generate
  during the pipeline execution: frame, cell, nucleus, cytoplasm, ...


## References :

Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation (2015).
