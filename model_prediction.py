"""
TESTING MODULE: Set inputs (1) and test the model (2).

The module works with 3D grayscale images for super-resolution tasks,
using U-Net, SR-CNN, EDSR, WDSRa, WDSRb (CNNs) 
and pix2pix, CycleGAN (GANs) models.
It reads images in "original_imgs" folder and models in "training" folder, 
and saves results in "testing" folder. Image shold be preprocessed as for training. 


NOTES:
Notes on Modality [M] -> Saved results:
[0] Prediction on unseen slice interval or volume (for 2D and 3D models)
    -> predicted 3D image or slice interval (.tif).
[1] Prediction on validation patches (for 2D and 3D) 
    -> figure (.tif) for predicted patch and its relative metrics.
[2] Prediction on validation slices (for 2D models)
    -> figure (.tif) for predicted slice and its relative metrics.
[3] Prediction on multiple unseen slices/volumes using the same model (for 2D and 3D models)
    (images located in the folder: "original_imgs/for_multiple_prediction)".
    -> predicted 3D images or slice intervals (.tif).
[4,5,6] All the models in the "training" folder are iteratively used to predict 
    unseen data [4], patches [5] or slicees [6] (as in [0],[1],[2], respectively).

Notes on inputs:
-LR_NAME/HR_NAME: (String) Name and format of images stored in "original_imgs" folder.
    HR required only for Modality(M)[1] and M[2]. HR should be equal to False in M[0].
-MODEL_NAME (String) Model name and format, stored in "training" folder. Unuseful in M[4,5,6].
-SCALING (Integer/Float) Scaling factor between LR and HR. Required for M[0]. 
    - For "resnet_group" is mandatory and must be an integer. 
    - In "unet"/"pix2pix"/"cyclegan" just refers to potential bicubic interpolation and can be a float.
-MODALITY (Integer, 0-6) See above.
-SLICE_INT (False or Integer (A,B)) Slice interval (from A to B) for prediction. 
    If False all the slices are used. For 3D models should be higher than PATCH_SIZE. 
-BICUBIC: (Boolean) Required for "unet", "pix2pix", "cyclegan" when LR is not upscaled to HR shape.
-PATCH_SIZE: (Integer) Size of patches (pixels) of the model. For "resnet_group" refers to LR.
-OVERLAP: (Float, 0-0.99) Fraction of patch overlapping of the model. From[0]: no overlapping, 
    to [1]: complete overlapping (not allowed). Allowed only for patches in M[1,5].
-NORM: (Integer, 0-4) Image normalization method. [0] no normalization (not to train), 
    [1] tf.keras.utils.normalize, [2] in the range (0-1), [3] in the range (-1,1).
-CHANNELS: (Integer, 1 or 3) [1] grayscale images, [3] tripled grayscale.
-TEST_SIZE: (Float, 0-1) Fraction of validation (vs training) data.
    For "cyclegan" on unpaired images it should be minimized (e.g., 1e-50).
-VAL_SLICES: (False or Integer) Slice interval for validation data. False = random split.
-MODEL_GROUP: (String) "unet", "resnet_group" (srresnet, edsr, wdsra, wdsrb), "pix2pix",
    "cyclegan".
-MODEL: (String) "MODEL_Xd" (MODEL: unet, srresnet, edsr, wdsra, wdsrb, pix2pix,
     cyclegan), (Xd: 2d or 3d) (e.g., "unet_2d", "unet_3d", "pix2pix_2d").
-BATCH_SIZE: (Integer) Number of batches used to train the model.
-NUMBER: (False or Integers (A,B,C)). Default: False. True allows to  predict 
    on a fix patch/slice during/after training, selected choosing: 
    A: number of patch, B: number of 3D patch slice, C: number of 3D image slice.
    2D models do not consider B, 3D models do not consider C.   
    [C] in range (0, Z), [B] in range (0, PATCH_SIZE), [A] (for overlap=0) from 0 to:
    - ((X//PATCH_SIZE)*(Y//PATCH_SIZE)*Z)*(TEST_SIZE) [2D],
    - ((X//PATCH_SIZE)*(Y//PATCH_SIZE)*(Z//PATCH_SIZE))*(TEST_SIZE) [3D].
-FORCE_PRED: (Boolean) If True, it allows to force unseen prediction ([0,3,4])
    for 2D methods, with a better but slower slice allocation.  
    
Final note on "pix2pix" and "cyclegan": 
These GANs are written to work on several, different image-to-image translation tasks. 
Thus, parameters set by the authors (pix2pix: Isola et al., 2016; cyclegan: Zhu et al., 2017) 
are directly implemented.
They works only with:
- PATCH_SIZE of 256/128/64/32 for "pix2pix" and 256/128 for "cyclegan" (both in 2D and 3D)
- NORM=3, BATCH_SIZE=1
- FORCE_PRED=True (prediction with a loop for each batch, with BATCH_SIZE=1 imposed)
If "cyclegan" are trained on unpaired images, metrics make no physical sense.                                                   
"""



# 1. SET INPUTS
## 1a. Image and model name
LR_NAME = "LR_image_cubic.tif" 
HR_NAME = 0
MODEL_NAME = "1_model.hdf5"
SCALING=4

## 1b. Prediction parametrs
MODALITY=0
SLICE_INT=False 

## 1c. Preprocessing
BICUBIC=False   
PATCH_SIZE=128  
OVERLAP=0     
NORM=3          
CHANNELS=1    
TEST_SIZE=0.25
VAL_SLICES=False  

## 1d. Details on trained model
MODEL_GROUP="pix2pix" 
MODEL="pix2pix_2d"                      
BATCH_SIZE=1 

## 1e. Supplementary options
NUMBER=False 
FORCE_PRED=True
                           
       
        
# 2. "RUN FILE" TO RUN THE (FOLLOWING) CODE











###############################################################################
###############################################################################
# MAIN CODE FOR PREDICTION


# A. Folder for predictions
import os, shutil 
if os.path.exists("prediction"):
    shutil.rmtree("prediction")
os.makedirs("prediction")






# B. Prediction in different modalities

from tensorflow.keras.models import load_model
DIMENSIONS = MODEL.split("_")[1]

if MODEL_GROUP=="pix2pix":  
    NORM=3
    FORCE_PRED=True
    if PATCH_SIZE!=256 and PATCH_SIZE!=128 and PATCH_SIZE!=64 and PATCH_SIZE!=32:
        raise ValueError("PATCH_SIZE in pix2pix can be only: 256, 128, 64, 32")
        
if MODEL_GROUP=="cyclegan":  
    NORM=3
    FORCE_PRED=True
    if PATCH_SIZE!=256 and PATCH_SIZE!=128:
        raise ValueError("PATCH_SIZE in cyclegan can be only: 256, 128")


## B.1. Modality 0,1,2,3 (single model)
if MODALITY!=4 or MODALITY!=5 or MODALITY!=6:
    import tensorflow as tf
    import tensorflow_addons as tfa
    model = load_model("training/"+MODEL_NAME, compile=False, custom_objects={"tf": tf, "tfa": tfa})



### [0] Prediction on unseen slice interval or volume (for 2D and 3D)
    if MODALITY==0:
        print("Modality [0]: Unseen data prediction")
        from utils import pred_unseen
        imgSR, fig = pred_unseen(model, LR_name=LR_NAME, bicubic=BICUBIC, scaling=SCALING, 
                                 patch_size=PATCH_SIZE, overlap=0, norm=NORM, 
                                 channels=CHANNELS, dimensions=DIMENSIONS, slice_int=SLICE_INT,
                                 model_group=MODEL_GROUP, force_pred=FORCE_PRED)
        from skimage import io
        io.imsave("prediction/predicted_unseen.tif", imgSR)
        fig.savefig("prediction/predicted_unseen_slice_fig.tif")


           
### [1] Prediction on validation patches (for 2D and 3D)          
    elif MODALITY==1:
        print("Modality [1]: Patch prediction")
        from utils import img_prep
        X_train, X_test, y_train, y_test = img_prep(LR_NAME, HR_NAME, bicubic=BICUBIC, 
                                                    dimensions=DIMENSIONS, test_size=TEST_SIZE, 
                                                    patch_size=PATCH_SIZE, overlap=OVERLAP, 
                                                    norm=NORM, channels=CHANNELS, 
                                                    model_group=MODEL_GROUP, show=False,
                                                    val_slices=VAL_SLICES)

        from utils import pred_on_patch
        fig, metrics = pred_on_patch(model, X_test, y_test, dimensions=DIMENSIONS, number=NUMBER)
        fig.savefig("prediction/predicted_patch.tif")
        with open("prediction/predicted_patch_metrics.txt", 'w') as f:
            f.write("psnr, ssim, mse: %f, %f, %f" % (metrics))
        print("Figure and metrics for patch prediction saved")



### [2] Prediction on validation slices (for 2D)
    elif MODALITY==2:
        print("Modality [2]: Slice prediction (2D)")
        if DIMENSIONS=="3d":
            print("Modality [2] not allowed for 3D images")
        elif DIMENSIONS=="2d":
            from utils import pred_on_slice
            fig, metrics = pred_on_slice(model, LR_name=LR_NAME, HR_name=HR_NAME, 
                                         dimensions=DIMENSIONS, test_size=TEST_SIZE, 
                                         patch_size=PATCH_SIZE, overlap=0, norm=NORM, 
                                         channels=CHANNELS, bicubic=BICUBIC, number=NUMBER,
                                         model_group=MODEL_GROUP, val_slices=VAL_SLICES)
            fig.savefig("prediction/predicted_slice.tif")
            with open("prediction/predicted_slice_metrics.txt", 'w') as f:
                f.write("psnr, ssim, mse: %f, %f, %f" % (metrics))
            print("Figure and metrics for slice prediction saved")
        else:
            raise ValueError("Error in: MODEL")
        
  
    
### [3] Prediction on multiple unseen slices/volumes using the same model (for 2D and 3D models)
    elif MODALITY==3:
        print("Modality [3]: Multiple unseen data prediction")
        from glob import glob
        from utils import pred_unseen
        for i in enumerate(glob("original_imgs/for_multiple_prediction\\*")):
            LR_NAME = i[1].split("original_imgs/")[1]
            imgSR, fig = pred_unseen(model, LR_name=LR_NAME, bicubic=BICUBIC, scaling=SCALING, 
                                     patch_size=PATCH_SIZE, overlap=0, norm=NORM, 
                                     channels=CHANNELS, dimensions=DIMENSIONS, 
                                     slice_int=SLICE_INT, model_group=MODEL_GROUP,
                                     force_pred=FORCE_PRED)
            from skimage import io
            io.imsave("prediction/predicted_unseen_%s.tif" % i[0], imgSR)
            fig.savefig("prediction/predicted_unseen_slice_fig_%s.tif" % i[0])




## B2. Modality 4,5,6 (multiple models)
elif MODALITY==4 or MODALITY==5 or MODALITY==6:
    from glob import glob
    for i in enumerate(glob("training/*.hdf5")):
        import tensorflow as tf
        import tensorflow_addons as tfa
        model = load_model(i[1], compile=False, custom_objects={"tf": tf, "tfa": tfa})


        
### [4] All the models in the "training" folder are iteratively used to predict unseen data        
        if MODALITY==4:
            print("Modality [4]: Unseen data prediction, using more models")
            from utils import pred_unseen
            imgSR, fig = pred_unseen(model, LR_name=LR_NAME, bicubic=BICUBIC, scaling=SCALING, 
                                     patch_size=PATCH_SIZE, overlap=0, norm=NORM, 
                                     channels=CHANNELS, dimensions=DIMENSIONS, 
                                     slice_int=SLICE_INT, model_group=MODEL_GROUP,
                                     force_pred=FORCE_PRED)
            from skimage import io
            io.imsave("prediction/predicted_unseen_%s.tif" % i[0], imgSR)
            fig.savefig("prediction/predicted_unseen_slice_fig_%s.tif" % i[0])
   
        
   
### [5] All the models in the "training" folder are iteratively used to predict patch         
        elif MODALITY==5:
            print("Modality [5]: Patch prediction, using more models")
            from utils import img_prep
            X_train, X_test, y_train, y_test = img_prep(LR_NAME, HR_NAME, bicubic=BICUBIC, 
                                                        dimensions=DIMENSIONS, test_size=TEST_SIZE, 
                                                        patch_size=PATCH_SIZE, overlap=OVERLAP, 
                                                        norm=NORM, channels=CHANNELS, 
                                                        model_group=MODEL_GROUP, show=False,
                                                        val_slices=VAL_SLICES)

            from utils import pred_on_patch
            fig, metrics = pred_on_patch(model, X_test, y_test, dimensions=DIMENSIONS, number=NUMBER)
            fig.savefig("prediction/predicted_patch_%s.tif" % i[0])
            with open("prediction/predicted_patch_metrics_%s.txt" % i[0], 'w') as f:
                f.write("psnr, ssim, mse: %f, %f, %f" % (metrics))
            print("Figure and metrics for patch prediction saved")
      
        
      
### [6] All the models in the "training" folder are iteratively used to predict slice        
        elif MODALITY==6:
            print("Modality [6]: Slice prediction (2D), using more models")
            if DIMENSIONS=="3d":
                print("Modality [6] not allowed for 3D images")
            elif DIMENSIONS=="2d":
                from utils import pred_on_slice
                fig, metrics = pred_on_slice(model, LR_name=LR_NAME, HR_name=HR_NAME, 
                                             dimensions=DIMENSIONS, test_size=TEST_SIZE, 
                                             patch_size=PATCH_SIZE, overlap=0, norm=NORM, 
                                             channels=CHANNELS, bicubic=BICUBIC, number=NUMBER,
                                             model_group=MODEL_GROUP, val_slices=VAL_SLICES)
                fig.savefig("prediction/predicted_slice_%s.tif" % i[0])
                with open("prediction/predicted_slice_metrics_%s.txt" % i[0], 'w') as f:
                    f.write("psnr, ssim, mse: %f, %f, %f" % (metrics))
                print("Figure and metrics for slice prediction saved")
            else:
                raise ValueError("Error in: MODEL")

###############################################################################
###############################################################################
            

