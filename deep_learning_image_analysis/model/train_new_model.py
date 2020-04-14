from model import unet_arch as unet
from run_parameters import training_data_path, seg_data_path
import numpy as np_
import itk
from tensorflow import keras
import h5py


def Import_data(frames_path, masks_path):
    
    print("Import raw frames ...")
    
    # Read input image
    itk_image = itk.imread(training_data_path)
    raw_frames = itk.array_from_image(itk_image).astype(np_.uint8)
    
    
    print("Import masks ...")
    
    with h5py.File(masks_path, 'r') as f:
        # List all groups
        print("Keys: %s" % f.keys())
        a_group_key = list(f.keys())[0]
    
        # Get the data
        data_ = list(f[a_group_key])
    
    masks=[]
    
    for mask in data_:
    
        masks.append(np_.reshape(mask-1,(mask.shape)))
        
    return raw_frames, masks




def Train_model(channel:int, model_name: str):
    
    raw_frames, masks = Import_data(training_data_path, seg_data_path)
    
    frame_shape=raw_frames[0].shape
    
    model= unet.get_unet(frame_shape[0],frame_shape[1],channel)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    print("fit model ...")
    
    checkpointer = keras.callbacks.ModelCheckpoint(model_name+'.h5', verbose=1, save_best_only=True)
    callbacks = [keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
                 keras.callbacks.TensorBoard(log_dir='logs')]
    history=model.fit(
      raw_frames,
      masks,
      validation_split=0.2, batch_size=10, epochs=30, callbacks=callbacks)
    
    model.save_weights(model_name+".h5")
    
    
    
    
    
    
    
    
    
    
    
    
    