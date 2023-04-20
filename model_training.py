"""
TRAINING MODULE: Set inputs (1) and train the model (2).

The module works with 3D grayscale images (or  stacks) for super-resolution tasks,
using U-Net, SR-CNN, EDSR, WDSRa, WDSRb (CNNs) 
and pix2pix, CycleGAN (GANs) models.
It reads images in "original_imgs" folder, and saves results in "training" folder. 


NOTES:
Notes on saved results: 
1. Trained model (.hdf5), 
2. Training/validation loss and metrics (.csv) and plots (.tif), 
3. Training time in h:m:s (.txt),
4. Final predicted patch and slice (figure, .tif) and metrics (print) (only if SAVE_FIG=True),
5. Intermediate models and predicted patches each selected epochs (if SAVE_INT=True)

Notes on inputs:
-LR_NAME/HR_NAME: (String) Name and format of images stored in "original_imgs" folder.
-BICUBIC: (Boolean) Required for "unet", "pix2pix", "cyclegan" when LR is not upscaled to HR shape.
-PATCH_SIZE: (Integer) Size of patches (pixels). For "resnet_group" refers to LR.
-OVERLAP: (Float, 0-0.99) Fraction of patch overlapping. From [0]: no overlapping, 
    to [1]: complete overlapping (not allowed).
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
-NUM_FILTERS: (Integer) Number of filters to start the network.
-OUT_ACTIVATION: Output activation function from:
    https://www.tensorflow.org/api_docs/python/tf/keras/activations
-SCALING (Integer) Scaling factor between LR and HR. 
    It is not required for "unet"/"pix2pix"/"cyclegan" (is 1x). 
    It can be only 2x, 3x, 4x for "resnet_group".
NUM_RES_BLOCKS: (Integer) Number of residual blocks. Only for "resnet_group".
RES_BLOCK_EXP: (Integer) Expansion of redidual blocks. Only for wdsr. 
    It is suggested to be between 2x and 4x for wdsra, and between 6x and 9x for wdsrb.  
-LOSS: https://www.tensorflow.org/api_docs/python/tf/keras/losses
-OPTIMIZER: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers 
-METRICS: https://www.tensorflow.org/api_docs/python/tf/keras/metrics 
    Peak signal-to-noise ratio is also available typing "PSNR", if NORM=[2] or [3].
-EPOCHS: (Integer) Number of epochs for training.
-BATCH_SIZE: (Integer) Number of batches for training.
-SHUFFLE: https://www.tensorflow.org/api_docs/python/tf/keras/Model
-SAVE_INT: (Integer, 0-EPOCHS) Default: SAVE_INT=EPOCHS, (no intermediate saving)
    otherwise it saves additional intermediate models/predicted patches each choosen epoch.
-SAVE_FIG: (Boolean) If True it also saves final predicted patch/slice and metrics.
-CUSTOM_FUN: (Boolean) Default: False. True allows to add functions in "utils_custom_fun.py".
-NUMBER: (False or Integers (A,B,C)). Default: False. True allows to  predict 
    on a fix patch/slice during/after training, selected choosing: 
    A: number of patch, B: number of 3D patch slice, C: number of 3D image slice.
    2D models do not consider B, 3D models do not consider C.   
    C in range (0, Z), B in range (0, PATCH_SIZE), A (for overlap=0) from 0 to:
    - ((X//PATCH_SIZE)*(Y//PATCH_SIZE)*Z)*(TEST_SIZE) [2D],
    - ((X//PATCH_SIZE)*(Y//PATCH_SIZE)*(Z//PATCH_SIZE))*(TEST_SIZE) [3D]. 

Final note on "pix2pix" and "cyclegan": 
These GANs are written to work on several, different image-to-image translation tasks. 
Thus, parameters set by the authors (pix2pix: Isola et al., 2016; cyclegan: Zhu et al., 2017) 
are directly implemented.
They works only with: 
- PATCH_SIZE of 256/128/64/32 for "pix2pix" and 256/128 for "cyclegan" (both in 2D and 3D), NORM=3
- For "Network": NUM_FILTERS=64; OUT_ACTIVATION="tanh" (for generator)
- For "Compile": parameters cannot be selected (are implemented following the paper)
- For "Training": BATCH_SIZE=1 and SHUFFLE=True
If "cyclegan" are trained on unpaired images, metrics make no physical sense.                                                   
"""



# 1. SET INPUTS
## 1a. Image name
LR_NAME = "LR_image_cubic.tif" 
HR_NAME = "HR_image.tif"

## 1b. Preprocessing
BICUBIC=False   
PATCH_SIZE=128
OVERLAP=0     
NORM=3                        # "pix2pix" and "cyclegan" (=3)    
CHANNELS=1 
TEST_SIZE=0.25
VAL_SLICES=False

## 1c. Training parametrs
### - Network
MODEL_GROUP="pix2pix"           
MODEL="pix2pix_2d"
NUM_FILTERS=64                # no in "pix2pix" and "cyclegan" (=64)
OUT_ACTIVATION="tanh"         # no in "pix2pix" and "cyclegan" (="tanh")
SCALING=4                     # only for resnet_group - all
NUM_RES_BLOCKS=16             # only for resnet_group - all
RES_BLOCK_EXP=6               # only for resnet_group - wdsr

### - Compile                 # no in "pix2pix" and "cyclegan"
LOSS="mean_squared_error"
OPTIMIZER="adam"                   
METRICS="PSNR"

### - Training                     
EPOCHS=100                      
BATCH_SIZE=1                   # no in "pix2pix" and "cyclegan" (=1)
SHUFFLE=True                   # no in "pix2pix" and "cyclegan" (=True)         
SAVE_INT=EPOCHS   
SAVE_FIG=True    
                  
## 1d. Supplementary options
CUSTOM_FUN=False               # learning rate can be customized in custom_fun.py
NUMBER=False     


 
# 2. "RUN FILE" TO RUN THE (FOLLOWING) CODE










###############################################################################
###############################################################################
# MAIN CODE FOR TRAINING


# A. Loading and Preprocessing

DIMENSIONS, NETWORK = MODEL.split("_")[1], MODEL.split("_")[0]

from utils import img_prep
if MODEL_GROUP=="pix2pix" or MODEL_GROUP=="cyclegan":
    NORM=3
X_train, X_test, y_train, y_test = img_prep(LR_NAME, HR_NAME, bicubic=BICUBIC, 
                                             dimensions=DIMENSIONS, test_size=TEST_SIZE, 
                                             patch_size=PATCH_SIZE, overlap=OVERLAP, 
                                             norm=NORM, channels=CHANNELS,
                                             model_group=MODEL_GROUP, 
                                             val_slices=VAL_SLICES)

INPUT_SHAPE, N_CLASSES = X_train.shape[1:], X_train.shape[-1]






# B. Training: it defines, compiles and trains model, producing results

## B.1. CONVOLUTIONAL NEURAL NETWORS ("unet" and "resnet_group")

if MODEL_GROUP!="pix2pix" and MODEL_GROUP!="cyclegan":
    
    ### - Network
    build_model = getattr(__import__(MODEL_GROUP), MODEL)
    if MODEL_GROUP=="unet":
        model = build_model(INPUT_SHAPE, N_CLASSES, NUM_FILTERS, OUT_ACTIVATION)
    elif MODEL_GROUP=="resnet_group":
        if NETWORK=="srresnet" or NETWORK=="edsr":
            model = build_model(INPUT_SHAPE, N_CLASSES, SCALING, NUM_FILTERS, 
                                NUM_RES_BLOCKS, OUT_ACTIVATION)
        elif NETWORK=="wdsra" or NETWORK=="wdsrb":
            model = build_model(INPUT_SHAPE, N_CLASSES, SCALING, NUM_FILTERS, 
                                NUM_RES_BLOCKS, RES_BLOCK_EXP, OUT_ACTIVATION) 
        else:
            raise ValueError("Error in: MODEL")
                                     

    ### - Compile
    if METRICS=="PSNR":
        from utils import PSNR
        model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=[PSNR])    
    else:
        model.compile(loss=LOSS, optimizer=OPTIMIZER, metrics=METRICS)            
    print("Model compiled")                                                   


    ### - Training
    train = getattr(__import__(MODEL_GROUP), "train")    
    history = train(model, X_train, y_train, BATCH_SIZE, EPOCHS, X_test, y_test, 
                    save_int=SAVE_INT, accuracy=METRICS, dimensions=DIMENSIONS, 
                    number=NUMBER, shuffle=SHUFFLE, custom_fun=CUSTOM_FUN)




## B.2. GENERATIVE ADVERSIRIAL NETWORKS ("pix2pix", "cyclegan")

### "pix2pix"
elif MODEL_GROUP=="pix2pix": 
    define_generator = getattr(__import__(MODEL_GROUP), 
                               "define_generator"+str(PATCH_SIZE)+"_"+str(DIMENSIONS)) 
    define_discriminator = getattr(__import__(MODEL_GROUP), 
                                   "define_discriminator_"+str(DIMENSIONS))    
    define_gan = getattr(__import__(MODEL_GROUP), "define_gan_"+str(DIMENSIONS)) 
    train = getattr(__import__(MODEL_GROUP), "train")
    
    NORM=3
    OUT_ACTIVATION="tanh"
    if PATCH_SIZE!=256 and PATCH_SIZE!=128 and PATCH_SIZE!=64 and PATCH_SIZE!=32:
        raise ValueError("PATCH_SIZE in pix2pix can be only: 256, 128, 64, 32")

    import numpy as np
    D_MODEL = define_discriminator(INPUT_SHAPE)
    G_MODEL = define_generator(INPUT_SHAPE, N_CLASSES)
    GAN_MODEL = define_gan(G_MODEL, D_MODEL, INPUT_SHAPE)
    history = train(D_MODEL, G_MODEL, GAN_MODEL, 
                    dataset=np.array([X_train, y_train]), 
                    dataset_val=np.array([X_test, y_test]), 
                    epochs=EPOCHS, n_batch=1, n_patch=PATCH_SIZE//16, save_int=SAVE_INT,
                    dimensions=DIMENSIONS, number=NUMBER)  


### "cyclegan"
elif MODEL_GROUP=="cyclegan": 
    NORM=3
    OUT_ACTIVATION="tanh"
    
    if PATCH_SIZE!=256 and PATCH_SIZE!=128:
        raise ValueError("PATCH_SIZE in cyclegan can be only: 256, 128")

    if PATCH_SIZE==256:
        N_RESNET=9
    elif PATCH_SIZE==128:
        N_RESNET=6
    
    define_generator = getattr(__import__(MODEL_GROUP), "define_generator_"+str(MODEL)) 
    g_model_AtoB = define_generator(INPUT_SHAPE, N_CLASSES, N_RESNET, OUT_ACTIVATION)
    g_model_BtoA = define_generator(INPUT_SHAPE, N_CLASSES, N_RESNET, OUT_ACTIVATION)
    
    define_discriminator = getattr(__import__(MODEL_GROUP), "define_discriminator_"+str(DIMENSIONS)) 
    d_model_A = define_discriminator(INPUT_SHAPE)
    d_model_B = define_discriminator(INPUT_SHAPE)

    define_composite_model = getattr(__import__(MODEL_GROUP), "define_composite_model_"+str(DIMENSIONS)) 
    c_model_AtoB = define_composite_model(g_model_AtoB, d_model_B, g_model_BtoA, INPUT_SHAPE)
    c_model_BtoA = define_composite_model(g_model_BtoA, d_model_A, g_model_AtoB, INPUT_SHAPE)

    train = getattr(__import__(MODEL_GROUP), "train")    
    import numpy as np
    history = train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, 
                    dataset=np.array([X_train, y_train]), dataset_val=np.array([X_test, y_test]),
                    epochs=EPOCHS, dimensions=DIMENSIONS, save_int=SAVE_INT, number=NUMBER)






# C. Prediction on a patch and slice: visual check

if MODEL_GROUP=="pix2pix":  
    model=G_MODEL
    NORM=3
    
elif MODEL_GROUP=="cyclegan":  
    model=g_model_AtoB
    NORM=3


## Patch
from utils import pred_on_patch
figA,_ = pred_on_patch(model, X_test, y_test, dimensions=DIMENSIONS, number=NUMBER)
if SAVE_FIG==True:
        figA.savefig("training/4_predicted_patch.tif")


## Slice
if DIMENSIONS=="3d":
    print("Prediction on slice is not allowed for 3D models")
elif DIMENSIONS=="2d":
    from utils import pred_on_slice
    figB,_ = pred_on_slice(model, LR_name=LR_NAME, HR_name=HR_NAME, 
                           dimensions=DIMENSIONS, test_size=TEST_SIZE, 
                           patch_size=PATCH_SIZE, overlap=0, norm=NORM, 
                           channels=CHANNELS, bicubic=BICUBIC, number=NUMBER,
                           model_group=MODEL_GROUP, val_slices=VAL_SLICES)
    if SAVE_FIG==True:
        figB.savefig("training/4_predicted_slice.tif")

###############################################################################
###############################################################################