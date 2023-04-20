""" 
SR-ResNet, EDSR, WDSR (WDSRa, WDSRb)

Ledig et al. (2017): https://arxiv.org/abs/1609.04802 
Lim et al. (2017): https://arxiv.org/abs/1707.02921
Yu et al. (2018): https://arxiv.org/abs/1808.08718 

Further information in:
He et al. (2015): https://arxiv.org/abs/1512.03385
Shi et al. (2016): https://arxiv.org/abs/1609.05158
https://keras.io/examples/vision/super_resolution_sub_pixel/
https://www.tensorflow.org/api_docs/python/tf/nn/depth_to_space
Salimans and Kingma (2016): https://arxiv.org/abs/1602.07868
https://www.tensorflow.org/addons/api_docs/python/tfa/layers/WeightNormalization
http://krasserm.github.io/2019/09/04/super-resolution/
https://github.com/yingDaWang-UNSW/EDSRGAN-3D
https://github.com/sanghyun-son/EDSR-PyTorch
https://github.com/JiahuiYu/wdsr_ntire2018
"""


# Libraries 
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, Conv3D
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import PReLU, ReLU
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import UpSampling3D
from tensorflow.keras.models import Model
import tensorflow as tf
import tensorflow_addons as tfa
from datetime import datetime
from tensorflow.keras.callbacks import CSVLogger, Callback
import os, shutil 
import matplotlib.pyplot as plt





# (1) Shared functions
# Sub-pixel convolution (for: srresnet_2d, edsr_2d, wdsr_2d)
def subpixel_conv2d(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)





# (2) 2D models


# 2.1. 2D SR-ResNet

def res_block_srresnet_2d(x_in, num_filters, momentum=0.8):
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x_in)
    x = BatchNormalization(momentum=momentum)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = Add()([x_in, x])
    return x    



def upsample_srresnet_2d(x, scale, num_filters):
    # 2D upsampling for scaling factor 2x, 3x, 4x
    # Adapted from Lim et al. (2017, EDSR). Ledig et al. (2017, SR-ResNet) just refer to 4x
    
    def upsample_srresnet_2d_sub(x, factor):
        x = Conv2D(num_filters * (factor ** 2), kernel_size=3, padding='same')(x)
        x = Lambda(subpixel_conv2d(scale=factor))(x)
        return PReLU(shared_axes=[1, 2])(x)

    if scale==2:
        x = upsample_srresnet_2d_sub(x, 2)
    elif scale==3:
        x = upsample_srresnet_2d_sub(x, 3)
    elif scale==4:
        x = upsample_srresnet_2d_sub(x, 2)
        x = upsample_srresnet_2d_sub(x, 2)
    else:
        raise ValueError("For srreset and edsr only SCALING of 2x, 3x and 4x is available")
        
    return x



def srresnet_2d(input_shape, n_classes, scale, num_filters=64, num_res_blocks=16, 
                activation="tanh"):
    inputs = Input(input_shape)
    
    x = Conv2D(num_filters, kernel_size=9, padding='same')(inputs)
    x = x_1 = PReLU(shared_axes=[1, 2])(x)

    for i in range(num_res_blocks):
        x = res_block_srresnet_2d(x, num_filters)

    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x_1, x])

    x = upsample_srresnet_2d(x, scale, num_filters)
    
    outputs = Conv2D(n_classes, kernel_size=9, padding='same', activation=activation)(x)

    model = Model(inputs, outputs, name="SR-ResNet_2D")
    model.summary()
    print("Output activation: %s" % activation)

    return model 




# 2.2. 2D EDSR

def res_block_edsr_2d(x_in, filters):
    x = Conv2D(filters, kernel_size=3, padding='same')(x_in)
    x = ReLU()(x)   
    x = Conv2D(filters, kernel_size=3, padding='same')(x)
    x = Add()([x_in, x])
    return x



def upsample_edsr_2d(x, scale, num_filters):
    
    def upsample_edsr_2d_sub(x, factor):
        x = Conv2D(num_filters * (factor ** 2), kernel_size=3, padding='same')(x)
        return Lambda(subpixel_conv2d(scale=factor))(x)

    if scale==2:
        x = upsample_edsr_2d_sub(x, 2)
    elif scale==3:
        x = upsample_edsr_2d_sub(x, 3)
    elif scale==4:
        x = upsample_edsr_2d_sub(x, 2)
        x = upsample_edsr_2d_sub(x, 2)
    else:
        raise ValueError("For SR-ResNet and EDSR only SCALING of 2x, 3x and 4x is available")
        
    return x



def edsr_2d(input_shape, n_classes, scale, num_filters=64, num_res_blocks=8,
            activation="tanh"):
    inputs = Input(input_shape)

    x = x_1 = Conv2D(num_filters, kernel_size=3, padding='same')(inputs)
    
    for i in range(num_res_blocks):
        x = res_block_edsr_2d(x, num_filters)
    
    x = Conv2D(num_filters, kernel_size=3, padding='same')(x)
    x = Add()([x_1, x])

    x = upsample_edsr_2d(x, scale, num_filters)
    
    outputs = Conv2D(n_classes, kernel_size=3, padding='same', activation=activation)(x)    

    model = Model(inputs, outputs, name="EDSR_2D")
    model.summary()
    print("Output activation: %s" % activation)

    return model




# 2.3. 2D WDSRa and 2D WDSRb
    
def conv2d_weightnorm(filters, kernel_size, padding='same', activation=None):
    return tfa.layers.WeightNormalization(Conv2D(filters, kernel_size, padding=padding, 
                                                 activation=activation), data_init=False)



def res_block_wdsr_a_2d(x_in, num_filters, expansion, kernel_size):
    x = conv2d_weightnorm(num_filters * expansion, kernel_size, padding='same')(x_in)
    x = ReLU()(x)
    x = conv2d_weightnorm(num_filters, kernel_size, padding='same')(x)
    x = Add()([x_in, x])
    return x


def res_block_wdsr_b_2d(x_in, num_filters, expansion, kernel_size):
    linear = 0.8
    x = conv2d_weightnorm(num_filters * expansion, kernel_size=1, padding='same')(x_in)
    x = ReLU()(x)
    x = conv2d_weightnorm(int(num_filters * expansion * linear), kernel_size=1, padding='same')(x)
    x = conv2d_weightnorm(num_filters, kernel_size, padding='same')(x)
    x = Add()([x_in, x])
    return x



def wdsr_2d(input_shape, n_classes, scale, num_filters, num_res_blocks, 
            res_block_expansion, res_block, activation):
    inputs = Input(input_shape)
    
    m = conv2d_weightnorm(num_filters, kernel_size=3, padding='same')(inputs)
    
    for i in range(num_res_blocks):
        m = res_block(m, num_filters, res_block_expansion, kernel_size=3)

    m = conv2d_weightnorm(n_classes * scale ** 2, kernel_size=3, padding='same')(m)
    m = Lambda(subpixel_conv2d(scale))(m)

    s = conv2d_weightnorm(n_classes * scale ** 2, kernel_size=5, padding='same')(inputs)
    s = Lambda(subpixel_conv2d(scale))(s)

    outputs = Add()([m, s])
    outputs = Activation(activation)(outputs)

    model = Model(inputs, outputs, name="WDSR_2D")
    model.summary()
    print("Output activation: %s" % activation)

    return model



def wdsra_2d(input_shape, n_classes, scale, num_filters=32, num_res_blocks=8, 
             res_block_expansion=4, activation="tanh"):
    return wdsr_2d(input_shape, n_classes, scale, num_filters, num_res_blocks, 
                   res_block_expansion, res_block=res_block_wdsr_a_2d, activation=activation)


def wdsrb_2d(input_shape, n_classes, scale, num_filters=32, num_res_blocks=8, 
             res_block_expansion=6, activation="tanh"):
    return wdsr_2d(input_shape, n_classes, scale, num_filters, num_res_blocks, 
                   res_block_expansion, res_block=res_block_wdsr_b_2d, activation=activation)





# (3) 3D models
# 3D implementation contains semplifications.  
# For all CNNs, sub-pixel convolution is replaced by: tensorflow.keras.layers.UpSampling3D,
# as tf.nn.depth_to_space works only in 2D. For WDSA also an additional (final) Conv3D
# is required to reshape to n_channels (kernel_size = 3 as in EDSR final Conv3D).


# 3.1. 3D SR-ResNet

def res_block_srresnet_3d(x_in, num_filters, momentum=0.8):
    x = Conv3D(num_filters, kernel_size=3, padding='same')(x_in)
    x = BatchNormalization(momentum=momentum)(x)
    x = PReLU(shared_axes=[1, 2, 3])(x)
    x = Conv3D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = Add()([x_in, x])
    return x    
    x = Conv3D(num_filters, kernel_size=3, padding='same')(x_in)
    x = BatchNormalization(momentum=momentum)(x)
    x = PReLU(shared_axes=[1, 2, 3])(x)
    x = Conv3D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization(momentum=momentum)(x)
    x = Add()([x_in, x])
    return x



def upsample_srresnet_3d(x, scale, num_filters):
    # 3D upsampling for scaling factor 2x, 3x, 4x
    # From Lim et al. (2017, EDSR). Ledig et al. (2017, SR-ResNet) just refer to 4x
    
    def upsample_srresnet_3d_sub(x, factor):
        x = Conv3D(num_filters * (factor ** 2), kernel_size=3, padding='same')(x)
        x = UpSampling3D(factor)(x)
        ###x = Lambda(subpixel_conv2d(scale=factor))(x)
        return PReLU(shared_axes=[1, 2, 3])(x)

    if scale==2:
        x = upsample_srresnet_3d_sub(x, 2)
    elif scale==3:
        x = upsample_srresnet_3d_sub(x, 3)
    elif scale==4:
        x = upsample_srresnet_3d_sub(x, 2)
        x = upsample_srresnet_3d_sub(x, 2)
    else:
        raise ValueError("For srreset and edsr only SCALING of 2x, 3x and 4x is available")
        
    return x



def srresnet_3d(input_shape, n_classes, scale, num_filters=64, num_res_blocks=16, 
                activation="tanh"):
    inputs = Input(input_shape)
    
    x = Conv3D(num_filters, kernel_size=9, padding='same')(inputs)
    x = x_1 = PReLU(shared_axes=[1, 2, 3])(x)

    for i in range(num_res_blocks):
        x = res_block_srresnet_3d(x, num_filters)

    x = Conv3D(num_filters, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x_1, x])

    x = upsample_srresnet_3d(x, scale, num_filters)
    
    outputs = Conv3D(n_classes, kernel_size=9, padding='same', activation=activation)(x)

    model = Model(inputs, outputs, name="SR-ResNet_3D")
    model.summary()
    print("Output activation: %s" % activation)

    return model 




# 3.2. 3D EDSR

def res_block_edsr_3d(x_in, filters):
    x = Conv3D(filters, kernel_size=3, padding='same')(x_in)
    x = ReLU()(x)   
    x = Conv3D(filters, kernel_size=3, padding='same')(x)
    x = Add()([x_in, x])
    return x



def upsample_edsr_3d(x, scale, num_filters):
    
    def upsample_edsr_3d_sub(x, factor):
        x = Conv3D(num_filters * (factor ** 2), kernel_size=3, padding='same')(x)
        ###return Lambda(subpixel_conv2d(scale=factor))(x)
        return UpSampling3D(factor)(x)

    if scale==2:
        x = upsample_edsr_3d_sub(x, 2)
    elif scale==3:
        x = upsample_edsr_3d_sub(x, 3)
    elif scale==4:
        x = upsample_edsr_3d_sub(x, 2)
        x = upsample_edsr_3d_sub(x, 2)
    else:
        raise ValueError("For SR-ResNet and EDSR only SCALING of 2x, 3x and 4x is available")
        
    return x



def edsr_3d(input_shape, n_classes, scale, num_filters=64, num_res_blocks=8,
            activation="tanh"):
    inputs = Input(input_shape)

    x = x_1 = Conv3D(num_filters, kernel_size=3, padding='same')(inputs)
    
    for i in range(num_res_blocks):
        x = res_block_edsr_3d(x, num_filters)
    
    x = Conv3D(num_filters, kernel_size=3, padding='same')(x)
    x = Add()([x_1, x])

    x = upsample_edsr_3d(x, scale, num_filters)
    
    outputs = Conv3D(n_classes, kernel_size=3, padding='same', activation=activation)(x)    

    model = Model(inputs, outputs, name="EDSR_3D")
    model.summary()
    print("Output activation: %s" % activation)

    return model




# 3.3. 3D WDSRa and 3D WDSRb
    
def conv3d_weightnorm(filters, kernel_size, padding='same', activation=None):
    return tfa.layers.WeightNormalization(Conv3D(filters, kernel_size, padding=padding, 
                                                 activation=activation), data_init=False)



def res_block_wdsr_a_3d(x_in, num_filters, expansion, kernel_size):
    x = conv3d_weightnorm(num_filters * expansion, kernel_size, padding='same')(x_in)
    x = ReLU()(x)
    x = conv3d_weightnorm(num_filters, kernel_size, padding='same')(x)
    x = Add()([x_in, x])
    return x


def res_block_wdsr_b_3d(x_in, num_filters, expansion, kernel_size):
    linear = 0.8
    x = conv3d_weightnorm(num_filters * expansion, kernel_size=1, padding='same')(x_in)
    x = ReLU()(x)
    x = conv3d_weightnorm(int(num_filters * expansion * linear), kernel_size=1, padding='same')(x)
    x = conv3d_weightnorm(num_filters, kernel_size, padding='same')(x)
    x = Add()([x_in, x])
    return x



def wdsr_3d(input_shape, n_classes, scale, num_filters, num_res_blocks, 
            res_block_expansion, res_block, activation):
    inputs = Input(input_shape)
    
    m = conv3d_weightnorm(num_filters, kernel_size=3, padding='same')(inputs)
    
    for i in range(num_res_blocks):
        m = res_block(m, num_filters, res_block_expansion, kernel_size=3)

    m = conv3d_weightnorm(n_classes * scale ** 2, kernel_size=3, padding='same')(m)
    ###m = Lambda(subpixel_conv2d(scale))(m)
    m = UpSampling3D(scale)(m)

    s = conv3d_weightnorm(n_classes * scale ** 2, kernel_size=5, padding='same')(inputs)
    ###s = Lambda(subpixel_conv2d(scale))(s)
    s = UpSampling3D(scale)(s)

    outputs = Add()([m, s])
    outputs = Conv3D(n_classes, kernel_size=3, padding='same')(outputs) 
    outputs = Activation(activation)(outputs)
 
    model = Model(inputs, outputs, name="WDSR_2D")
    model.summary()
    print("Output activation: %s" % activation)

    return model



def wdsra_3d(input_shape, n_classes, scale, num_filters=32, num_res_blocks=8, 
             res_block_expansion=4, activation="tanh"):
    return wdsr_3d(input_shape, n_classes, scale, num_filters, num_res_blocks, 
                   res_block_expansion, res_block=res_block_wdsr_a_3d, activation=activation)


def wdsrb_3d(input_shape, n_classes, scale, num_filters=32, num_res_blocks=8, 
             res_block_expansion=6, activation="tanh"):
    return wdsr_3d(input_shape, n_classes, scale, num_filters, num_res_blocks, 
                   res_block_expansion, res_block=res_block_wdsr_b_3d, activation=activation)





# (4) Training

# General training function
def train(model, X_train, y_train, batch_size, epochs, X_test, y_test, 
          save_int, accuracy, dimensions, number, show=False, shuffle=False, custom_fun=False):
    
    # - Initialization
    timer_start = datetime.now()
    
    if os.path.exists("training"):
        shutil.rmtree("training")
    os.makedirs("training")

    # - Callbacks
    csv_log = CSVLogger("training/2_loss_and_metrics.csv", 
                    separator=',', append=False)
    callbacks_list = [csv_log]
    
    if custom_fun==True:
        from utils_custom_fun import custom1_lr
        custom1_lr(callbacks_list)
        print("Customized function on [Learning Rate]: IN")
        
    if save_int!=epochs: 
        class CustomSaver(Callback):
            def __init__(self, X_test, y_test, dimensions, number, show): 
                self.X_test, self.y_test, self.model = X_test, y_test, model
                self.dimensions, self.number, self.show = dimensions, number, show
            def on_epoch_end(self, epoch, logs={}):
                if epoch in range(save_int-1,epochs,save_int):
                    self.model.save("training/model_epoch%d.hdf5" % epoch)
                    self.model.save_weights("training/model_weights_epoch%d.hdf5" % epoch)
                    from utils import pred_on_patch
                    fig,_ = pred_on_patch(self.model, self.X_test, self.y_test, 
                                        self.dimensions, self.number, self.show)
                    fig.savefig("training/pred_patch_epoch%d.tif" % epoch)
        callbacks_list.append(CustomSaver(X_test, y_test, dimensions, number, show))
        print('Model and predicted patch saving each %d epochs: IN' % save_int)
        
    # - Training
    history = model.fit(X_train, y_train, batch_size, epochs, 
                        validation_data=(X_test, y_test), 
                        verbose=1, shuffle=False, callbacks=callbacks_list)
    
    # - Save (1): model
    model.save("training/1_model.hdf5")
    model.save_weights("training/1_model_weights.hdf5")

    # - Save (2): loss and metric plots
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochsx = range(0, len(loss))
    fig1 = plt.figure(figsize=(7,5))
    plt.plot(epochsx, loss, "y", label="Training loss")
    plt.plot(epochsx, val_loss, "r", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    fig1.savefig("training/2_loss_plot.tif")
    acc = history.history[accuracy]
    val_acc = history.history["val_"+accuracy]
    fig2 = plt.figure(figsize=(7,5))
    plt.plot(epochsx, acc, "y", label="Training Accuracy")
    plt.plot(epochsx, val_acc, "r", label="Validation Accuracy")
    plt.title("Training and validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    fig2.savefig("training/2_metric_plot.tif")
    
    # - Save (3): training time    
    timer_end = datetime.now()
    execution_time = timer_end-timer_start
    with open("training/3_training_time.txt", 'w') as f:
        f.write(str(execution_time))
    print("Training completed in: ", execution_time)
    
    return history

