# Libraries
from skimage import io, img_as_uint
from datetime import datetime
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from patchify import patchify, unpatchify
from skimage.color import gray2rgb, rgb2gray
from tensorflow.keras.utils import normalize
from tensorflow.image import psnr
from skimage.metrics import peak_signal_noise_ratio as psnr_sk
from skimage.metrics import structural_similarity as ssim_sk
from skimage.metrics import mean_squared_error as mse_sk
from tqdm import tqdm





# (1) PREPROCESSING

# 1.1. Image reading
def img_read(LR_name, HR_name):
    imgLR = io.imread("original_imgs/"+LR_name)
    imgHR = io.imread("original_imgs/"+HR_name)
    return imgLR, imgHR
    



# 1.2. Bicubic interpolation
def bicubic_int(imgLR, imgHR):
    timer_start = datetime.now()
    print("Bicubic interpolation: IN")
    imgLR = img_as_uint(resize(imgLR, imgHR.shape, order=3))
    timer_end = datetime.now()
    execution_time = (timer_end-timer_start)
    print("Bicubic interpolation: OK (in: %s h:m:s)" % execution_time)
    return imgLR
    



# 1.3. Training/validation data splitting
def tra_val_split(imgLR, imgHR, test_size):
    LR_train, LR_test, HR_train, HR_test = train_test_split(imgLR, imgHR, 
                                                            test_size=test_size, 
                                                            random_state=0)
    return LR_train, LR_test, HR_train, HR_test
        



# 1.4. Main preprocessing function
def preprocessing(stack, patch_size, overlap, norm, channels, dimensions):
    
    # Patching
    patch_stack = []
    overlap_value = int(patch_size-(patch_size*overlap))
    if dimensions=="2d":
        patch_size_tuple=(patch_size, patch_size)
        for i in range(stack.shape[0]): 
            patches = patchify(stack[i], patch_size_tuple, step=overlap_value)
            patch_stack.append(patches)
        patch_stack = np.reshape(np.array(patch_stack), (-1,)+patch_size_tuple)
    elif dimensions=="3d":
        patch_size_tuple=(patch_size, patch_size, patch_size)
        patch_stack = patchify(stack, patch_size_tuple, step=overlap_value)
        patch_stack = np.reshape(patch_stack, (-1,)+patch_size_tuple)
    else:
        raise ValueError("Error in: MODEL")
                                                                                    
    # Normalization
    if norm==0:
        patch_stack = patch_stack
    elif norm==1:   
        patch_stack = normalize(patch_stack)
    elif norm==2:
        norm_factor = float(np.iinfo(patch_stack.dtype).max)
        patch_stack = patch_stack/norm_factor
    elif norm==3:
        norm_factor = float(np.iinfo(patch_stack.dtype).max)/2
        patch_stack = (patch_stack-norm_factor)/norm_factor
    else:
        raise ValueError("Error in: NORM")
       
    # Channels
    if channels==1:
        patch_stack = np.expand_dims(patch_stack, axis=-1)
    elif channels==3:
        patch_stack = gray2rgb(patch_stack)
    else:
        raise ValueError("Error in: CHANNELS")
    
    return patch_stack
         
        


# 1.5-Supplementary. Resizing function for LR in ResNet group (see 1.5., below)
def resnet_gr_fun(imgLR, imgHR):
    sc_fc = imgHR.shape[0]//imgLR.shape[0]
    if imgHR.shape[0]%imgLR.shape[0]==0:
        print("ResNet group. LR and HR image size: OK. Scaling factor: %dx" % sc_fc)
    else:
        # 1st way (raise error)
        raise ValueError("LR to HR scaling factor should be an integer")
    return imgLR




# 1.5. Complete reading and preprocessing 
# (In 2D: testing and validation data splitting is made on 2D slices, before 2D patching)
# (In 3D: testing and validation data splitting is made on 3D patches, after 3D patching)
def img_prep(LR_name, HR_name, dimensions, model_group, test_size=0.25, 
             patch_size=128, overlap=0, 
                norm=1, channels=1, val_slices=False, bicubic=False, show=True):
    
    # Image loading and upsampling
    imgLR, imgHR = img_read(LR_name, HR_name)
    if bicubic==True:
        imgLR = bicubic_int(imgLR, imgHR)
    else:
        print("Bicubic interpolation: not required")
    print("Image loading: OK")
     
    
    # Image size - model checkpoint
    if model_group!="resnet_group":
        if imgLR.shape[-1]!=imgHR.shape[-1]:
            raise ValueError('In "unet" or "pix2pix" or "cyclegan" LR and HR cannot have different sizes.')
    elif model_group=="resnet_group":
        print("OK")
        if imgLR.shape[-1]==imgHR.shape[-1]:
            raise ValueError('In "resnet_group" LR and HR cannot have the same size. They need a scaling factor.')
    else:
        raise ValueError("Error in: MODEL_GROUP")
        
    
    # Splitting and preprocessing
    # 2D preparation
    if dimensions=="2d":
        if model_group=="resnet_group":
            imgLR = resnet_gr_fun(imgLR, imgHR)
        sc_fc = imgHR.shape[0]//imgLR.shape[0]
        imgHR = imgHR[np.arange(0,imgHR.shape[0],sc_fc),:,:]
        # Splitting first
        if val_slices!=False:
            a, b, exc = val_slices[0], val_slices[1], [x for x in range(val_slices[0], val_slices[1])]       
            LR_train, LR_test, HR_train, HR_test = np.delete(imgLR, exc, 0), imgLR[a:b], np.delete(imgHR, exc, 0), imgHR[a:b]
            print("Validation percentage: %f" % (LR_test.shape[0]*100/imgLR.shape[0]))
        else:           
            LR_train, LR_test, HR_train, HR_test = tra_val_split(imgLR, imgHR, test_size)
            print("2D training and testing data splitting (%.3f): OK" % test_size)
        # Preprocessing then
        X_train = preprocessing(LR_train, patch_size, overlap, norm, channels, dimensions)
        X_test = preprocessing(LR_test, patch_size, overlap, norm, channels, dimensions)
        y_train = preprocessing(HR_train, patch_size*sc_fc, overlap, norm, channels, dimensions)
        y_test = preprocessing(HR_test, patch_size*sc_fc, overlap, norm, channels, dimensions)
            
    # 3D preparation
    if dimensions=="3d":
        if model_group=="resnet_group":
            imgLR = resnet_gr_fun(imgLR, imgHR)
        sc_fc = imgHR.shape[0]//imgLR.shape[0]
        
        if val_slices!=False:
            # Splitting first
            a, b = val_slices[0], val_slices[1]
            excLR, excHR = [x for x in range(a, b)], [x for x in range(a*sc_fc, b*sc_fc)]           
            LR_train, LR_test, HR_train, HR_test = np.delete(imgLR, excLR, 0), imgLR[a:b], np.delete(imgHR, excHR, 0), imgHR[a*sc_fc:b*sc_fc]
            print("Validation percentage: %f" % (LR_test.shape[0]*100/imgLR.shape[0])) 
            # Preprocessing then
            X_train = preprocessing(LR_train, patch_size, overlap, norm, channels, dimensions)
            X_test = preprocessing(LR_test, patch_size, overlap, norm, channels, dimensions)
            y_train = preprocessing(HR_train, patch_size*sc_fc, overlap, norm, channels, dimensions)
            y_test = preprocessing(HR_test, patch_size*sc_fc, overlap, norm, channels, dimensions)
        else:
            # Preprocessing first
            LR_patches = preprocessing(imgLR, patch_size, overlap, norm, channels, dimensions)
            HR_patches = preprocessing(imgHR, patch_size*sc_fc, overlap, norm, channels, dimensions)
            # Splitting then
            X_train, X_test, y_train, y_test = tra_val_split(LR_patches, HR_patches, test_size)
            print("3D training and testing data splitting (%.3f): OK" % test_size)


    # Print recap    
    print("Patching (%d, %d): OK. " % (patch_size, overlap), 
          "Residual pixels (for overlap=0): %d (0 in 2D), %d, %d in Z, Y, X" % 
          (imgLR.shape[0]-((imgLR.shape[0]//patch_size)*patch_size),
           imgLR.shape[1]-((imgLR.shape[1]//patch_size)*patch_size), 
           imgLR.shape[2]-((imgLR.shape[2]//patch_size)*patch_size))) 
    print("Normalization ([%d]): OK" % norm)
    print("Channels ([%d]): OK" % channels)
    
    
    # Visual check 
    print("Visual check:")
    numA, numB = np.random.randint(X_train.shape[0]), np.random.randint(X_test.shape[0])
    if dimensions=="2d":
        X_train_plt, y_train_plt = X_train[numA,:,:,0], y_train[numA,:,:,0]
        X_test_plt, y_test_plt = X_test[numB,:,:,0], y_test[numB,:,:,0]
    elif dimensions=="3d":
        numC = np.random.randint(X_train.shape[1])
        sc_fc = y_test.shape[-2]//X_test.shape[-2]
        X_train_plt, y_train_plt = X_train[numA,numC,:,:,0], y_train[numA,numC*sc_fc,:,:,0]
        X_test_plt, y_test_plt = X_test[numB,numC,:,:,0], y_test[numB,numC*sc_fc,:,:,0]
    plt.figure(figsize=(10, 10))
    plt.subplot(221)
    plt.imshow(X_train_plt, cmap="gray", interpolation=None)
    plt.title("LR train")
    plt.subplot(222)
    plt.imshow(y_train_plt, cmap="gray", interpolation=None)
    plt.title("HR train")
    plt.subplot(223)
    plt.imshow(X_test_plt, cmap="gray", interpolation=None)
    plt.title("LR test")
    plt.subplot(224)
    plt.imshow(y_test_plt, cmap="gray", interpolation=None)
    plt.title("HR test")
    if show==False:
        plt.close()
    else:
        plt.show()

    print('Preprocessing completed')
    return X_train, X_test, y_train, y_test





# (2) TRAINING 

# PSNR (Peak Signal-to-Noise Ratio) metric
def PSNR(y_true, y_pred):
    return psnr(y_true, y_pred, max_val=1)





# (3) VALIDATION/PREDICTION

# 3.1. Validation plot
def plot_validation(X_plt, y_plt, y_pred_plt, show=True, img_type=""):
    fig = plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.title("Testing LR %s" % img_type)
    plt.imshow(X_plt, cmap="gray", interpolation=None)
    plt.subplot(132)
    plt.title("Testing HR %s" % img_type)
    plt.imshow(y_plt, cmap="gray", interpolation=None)
    plt.subplot(133)
    plt.title("Predicted %s" % img_type)
    plt.imshow(y_pred_plt, cmap="gray", interpolation=None)
    if show==False:
        plt.close()
    else:
        plt.show()
    return fig




# 3.2. Validation metrics
def metrics_val(y_plt, y_pred_plt, show):
    y_plt, y_pred_plt = img_as_uint(y_plt)/((2**16)), img_as_uint(y_pred_plt)/((2**16))
    metrics = (psnr_sk(y_plt, y_pred_plt), 
               ssim_sk(y_plt, y_pred_plt), 
               mse_sk(y_plt, y_pred_plt))
    if show==True:
        print("psnr, ssim, mse: ", metrics)
    return metrics
    



# 3.3. Prediction on validation patch (2D and 3D) (training and validation data available)
def pred_on_patch(model, X_test, y_test, dimensions, number=False, show=True):
    numA, numB = np.random.randint(X_test.shape[0]), np.random.randint(X_test.shape[1])
    if number!=False:
        numA, numB = number[0], number[1]
        
    if dimensions=="2d":
        X_patch, y_patch = X_test[numA:numA+1], y_test[numA:numA+1]
        y_pred_patch=model.predict(X_patch)
        X_plt, y_plt, y_pred_plt = X_patch[0], y_patch[0], y_pred_patch[0]
    elif dimensions=="3d":
        X_patch, y_patch = X_test[numA:numA+1], y_test[numA:numA+1]
        y_pred_patch=model.predict(X_patch)
        sc_fc = y_test.shape[-2]//X_test.shape[-2]
        X_plt, y_plt, y_pred_plt = X_patch[0][numB], y_patch[0][numB*sc_fc], y_pred_patch[0][numB*sc_fc]
        
    if X_plt.shape[-1]==1: #channels
        X_plt, y_plt, y_pred_plt = np.squeeze(X_plt), np.squeeze(y_plt), np.squeeze(y_pred_plt)
    elif X_plt.shape[-1]==3:
        X_plt, y_plt, y_pred_plt = rgb2gray(X_plt), rgb2gray(y_plt), rgb2gray(y_pred_plt)
    
    if np.min(X_plt)<0 or np.min(y_plt)<0 or np.min(y_pred_plt)<0:
        X_plt, y_plt, y_pred_plt = (X_plt+1)/2.0, (y_plt+1)/2.0, (y_pred_plt+1)/2.0
    X_plt, y_plt, y_pred_plt = img_as_uint(X_plt), img_as_uint(y_plt), img_as_uint(y_pred_plt)
        
    fig = plot_validation(X_plt, y_plt, y_pred_plt, show, img_type="patch")
    metrics = metrics_val(y_plt, y_pred_plt, show)    
        
    return fig, metrics


 
 
# 3.4. Prediction on validation slice (only 2D) (training and validation data available)
def pred_on_slice(model, LR_name, HR_name, model_group, dimensions, test_size, patch_size, overlap, norm, channels, 
                  val_slices=False, bicubic=False, number=False, show=True):
    
    overlap=0
    
    if dimensions=="3d":
        print("Prediction on slice is not allowed for 3D models")
    
    elif dimensions=="2d":
        # Image loading and upsampling
        imgLR, imgHR = img_read(LR_name, HR_name)
        if bicubic==True:
            imgLR = bicubic_int(imgLR, imgHR)
     
        # Splitting 
        if model_group=="resnet_group":
            imgLR = resnet_gr_fun(imgLR, imgHR)
        sc_fc = imgHR.shape[0]//imgLR.shape[0]
        imgHR = imgHR[np.arange(0,imgHR.shape[0],sc_fc),:,:]
        if val_slices!=False:
            a, b, exc = val_slices[0], val_slices[1], [x for x in range(val_slices[0], val_slices[1])]       
            _, LR_test, _, HR_test = np.delete(imgLR, exc, 0), imgLR[a:b], np.delete(imgHR, exc, 0), imgHR[a:b]
        else:           
            _, LR_test, _, HR_test = tra_val_split(imgLR, imgHR, test_size)
            
        # Slice selection, preprocessing and prediction
        numC = np.random.randint(LR_test.shape[0])
        if number!=False:
            numC = number[2] 
        X_plt, y_plt = LR_test[numC], HR_test[numC] 
        pred_patches_pre = preprocessing(np.expand_dims(X_plt, 0), patch_size, overlap, norm, channels, dimensions) 
        
        pred_patches = []
        for i in range(pred_patches_pre.shape[0]):
            pred_patches_i = model.predict(np.expand_dims(pred_patches_pre[i],0))
            pred_patches.append(pred_patches_i)
        pred_patches = np.array(pred_patches)[:,0]
        if norm==3:
            pred_patches = (pred_patches + 1) / 2.0
        pred_patches = img_as_uint(pred_patches)
        
        pt_tuple = (imgLR.shape[1]//patch_size, imgLR.shape[2]//patch_size, patch_size*sc_fc, patch_size*sc_fc, channels)
        sl_tuple = (pt_tuple[0]*pt_tuple[2], pt_tuple[1]*pt_tuple[3])
        pred_patches = np.reshape(pred_patches, pt_tuple)
        if channels==1:
            pred_patches = np.squeeze(pred_patches)
        elif channels==3:
            pred_patches = rgb2gray(pred_patches)
        y_pred_plt = unpatchify(pred_patches, sl_tuple)
                
        X_plt, y_plt, y_pred_plt = X_plt[:sl_tuple[0], :sl_tuple[1]], y_plt[:sl_tuple[0], :sl_tuple[1]], y_pred_plt[:sl_tuple[0], :sl_tuple[1]]
        
        # Plot and metrics
        fig = plot_validation(X_plt, y_plt, y_pred_plt, show, img_type="slice")
        metrics = metrics_val(y_plt, y_pred_plt, show)
        
        return fig, metrics




# 3.5. Prediction on unseen data
def pred_unseen(model, LR_name, bicubic, scaling, patch_size, overlap, 
                norm, channels, dimensions, slice_int, model_group, force_pred):
    
    timer_start = datetime.now()
    print("Prediction starts")
    
    overlap=0
    
    # Image loading and upsampling
    imgLR = io.imread("original_imgs/"+LR_name)
    if bicubic==True:
        imgHR = np.zeros(int(imgLR.shape[0]*scaling), int(imgLR.shape[1]*scaling), int(imgLR.shape[2]*scaling))
        imgLR = bicubic_int(imgLR, imgHR)
    else:
        print("Bicubic interpolation: not required")
    print("Image loading: OK")
    
    # Scaling change
    if model_group!="resnet_group":
        scaling=1
    
    # Slice crop
    if slice_int!=False:
        imgLR_def = imgLR[slice_int[0]:slice_int[1]]
    else:
        imgLR_def = imgLR
    
    # Preprocessing
    LR_patches = preprocessing(imgLR_def, patch_size, overlap, norm, channels, dimensions) 
    print("Patching (%d, %d): OK. " % (patch_size, overlap), 
          "Residual pixels (for overlap=0): %d (0 in 2D), %d, %d in Z, Y, X" % 
          (imgLR.shape[0]-((imgLR.shape[0]//patch_size)*patch_size),
           imgLR.shape[1]-((imgLR.shape[1]//patch_size)*patch_size), 
           imgLR.shape[2]-((imgLR.shape[2]//patch_size)*patch_size)))
    
    # Prediction 
    if dimensions=="2d": 
        if force_pred==True:
            SR_patches = []
            print("Prediction of: %d 2D patches..." % LR_patches.shape[0])                                                            
            for i in tqdm(range(LR_patches.shape[0])):
                SR_patches_i = model.predict(np.expand_dims(LR_patches[i],0))
                SR_patches.append(SR_patches_i)
            SR_patches = np.array(SR_patches)[:,0]
        else:
            SR_patches = model.predict(LR_patches)
                
    # All-together prediction
    elif dimensions=="3d":
        SR_patches = [] 
        print("Prediction of: %d 3D patches..." % LR_patches.shape[0])                                                             
        for i in tqdm(range(LR_patches.shape[0])):
            SR_patches_i = model.predict(np.expand_dims(LR_patches[i],0))
            SR_patches.append(SR_patches_i)
        SR_patches = np.array(SR_patches)[:,0]
    
    if norm==3:
       SR_patches = (SR_patches + 1) / 2.0     
    SR_patches = img_as_uint(SR_patches)
    
    # Reshape
    # - Channels 
    if channels==1:
        imgSR_res = np.squeeze(SR_patches)
    elif channels==3:
        imgSR_res = img_as_uint(rgb2gray(SR_patches))       
    
    # - Dimensions 
    i0, i1, i2 = imgLR_def.shape[0]//patch_size, imgLR_def.shape[1]//patch_size, imgLR_def.shape[2]//patch_size
    if dimensions=="2d":
        pt_tuple = (-1, i1, i2, patch_size*scaling, patch_size*scaling)
        unp_tuple = (i1*patch_size*scaling, i2*patch_size*scaling)
        imgSR_res = np.reshape(imgSR_res, pt_tuple)      
        imgSR = []
        for i in range(imgSR_res.shape[0]):
            imgSR_i = unpatchify(imgSR_res[i], unp_tuple)
            imgSR.append(imgSR_i)
        imgSR = np.array(imgSR)   
    elif dimensions=="3d":
        pt_tuple = (-1, i1, i2, patch_size*scaling, patch_size*scaling, patch_size*scaling)
        unp_tuple = (i0*patch_size*scaling, i1*patch_size*scaling, i2*patch_size*scaling)   
        imgSR_res = np.reshape(imgSR_res, pt_tuple)
        imgSR = unpatchify(imgSR_res, unp_tuple)
        
    timer_end = datetime.now()
    execution_time = (timer_end-timer_start)
    print("Prediction ends (in: %s h:m:s). Predicted image saved." % execution_time) 
 
    # Slice plot
    numA = numB = np.random.randint(imgLR_def.shape[0])
    if dimensions=="3d":
        numB = numA*scaling
    fig = plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.title("Unseen LR slice")
    plt.imshow(imgLR_def[numA], cmap="gray", interpolation=None)
    plt.subplot(122)
    plt.title("Unseen SR slice")
    plt.imshow(imgSR[numB], cmap="gray", interpolation=None)
    plt.show()

    return imgSR, fig
    
 