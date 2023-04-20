# 3dSRCT
2D/3D CNN- and GAN-based super-resolution for 3D digital rocks



This repository is related to the following paper:  
Article Title: Exploring microstructure and petrophysical properties of microporous volcanic rocks through 3D multiscale and super-resolution imaging.
Authors: Gianmarco Buono, Stefano Caliro, Giovanni Macedonio, Vincenzo Allocca, Federico Gamba & Lucia Pappalardo.
Journal: Scientific Reports.
DOI: https://doi.org/10.1038/s41598-023-33687-x.



The repository contains codes to train models and predict images for super-resolution tasks. 
The implemented codes were mainly designed to work with 3D grayscale images 
(low resolution images, LR, as input and high resolution images, HR, as ground truth), 
using Convolutional Neural Networks (CNNs) and Generative Adversarial Networks (GANs). 
Particularly the following networks have been involved: 
- U-Net [1], SR-CNN [2], EDSR [3], WDSRa [4], WDSRb [4] (CNNs); 
- pix2pix [5], CycleGAN [6] (GANs).


The main modules are:
- "model_training": It allows to train models. 
  Input and ground truth images need to be loaded in a folder named "original_imgs"; 
  the code saves results in a folder named "training". 

- "model_prediction": It allows to predict super-resolution results. 
  Input image(s) and model(s) need to be loaded in folders named "original_imgs" 
  and "training", respectively; the code saves results in a folder named "testing". 

The modules just require to (i) set the inputs in the first lines and (ii) run the file. 
Detailed instructions and notes to set and run the codes are provided in the 
corresponding modules. 

They call additional modules contained in the repository. Particularly, the following modules 
provide functions to build and train 2D/3D models based on the selected inputs: 
"unet" (for U-Net), "resnet_group" (for SR-CNN, EDSR, WDSRa and WDSRb),
"pix2pix" (for pix2pix), "cyclegan" (for CycleGAN). 
The remaining modules (“utils” and “utils_custom_fun”) provide functions to preprocess data, 
optimize the training and predict results based on the selected inputs.

The codes are implemented using the following version of Tensorflow/Keras: 
Tensorflow 2.5.0, Python 3 (and originally designed to work in a Spyder IDE). 
Other libraries: datetime, glob, numpy (version: 1.19.5), matplotlib (3.4.3), 
os, pandas (1.3.4), patchify, random, shutil, skimage (0.18.3), sklearn (0.24.2), 
tensorflow_addons (0.16.1), tqdm (4.62.3).


[1] Ronneberger, O., Fischer, P. & Brox, T. 
U-Net: Convolutional Networks for Biomedical Image Segmentation in Medical Image Computing 
and Computer-Assisted Intervention – MICCAI 2015 (eds. Navab, N., Hornegger, J., Wells, W., 
Frangi, A.). Springer, pp. 234–241. https://doi.org/10.1007/978-3-319-24574-4_28 (2015).

[2] Ledig, C. et al. 
Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network. 
arXiv 2017, arXiv:1609.04802. https://doi.org/10.48550/arXiv.1609.04802 (2017).

[3] Lim, B., Son, S., Kim, H., Nah, S. & Lee, K.M. 
Enhanced Deep Residual Networks for Single Image Super-Resolution. 
arXiv 2017, arXiv:1707.02921. https://doi.org/10.48550/arXiv.1707.02921 (2017).

[4] Yu, J. et al. 
Wide Activation for Efficient and Accurate Image Super-Resolution. 
arXiv 2018, arXiv:1808.08718. https://doi.org/10.48550/arXiv.1808.08718 (2018). 

[5] Isola, P., Zhu, J.-Y., Zhou, T. & Efros, A.A. 
Image-to-Image Translation with Conditional Adversarial Networks. 
arXiv 2016, arXiv:1611.07004. https://doi.org/10.48550/arXiv.1611.07004 (2016).

[6] Zhu, J.-Y., Park, T., Isola, P., Efros, A.A.
Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.
arXiv 2017, arXiv:1703.10593. https://doi.org/10.48550/arXiv.1703.10593 (2017).
