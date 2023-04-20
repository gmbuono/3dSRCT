"""
U-Net

Ronneberger et al. (2015): https://arxiv.org/abs/1505.04597

Further information in:
https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/
https://github.com/bnsreenu/python_for_microscopists/blob/master/219-unet_model_with_functions_of_blocks.py
https://github.com/bnsreenu/python_for_microscopists/blob/master/204-207simple_unet_model.py
https://github.com/bnsreenu/python_for_image_processing_APEER/blob/master/tutorial122_3D_Unet.ipynb
"""


# Libraries
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, Conv3D 
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPool2D, MaxPool3D
from tensorflow.keras.layers import Conv2DTranspose, Conv3DTranspose 
from tensorflow.keras.layers import Concatenate
#from tensorflow.keras.layers import BatchNormalization
#from tensorflow.keras.layers import Dropout
#from tensorflow.keras.layers import UpSampling2D, UpSampling3D
from datetime import datetime
from tensorflow.keras.callbacks import CSVLogger, Callback
import os, shutil 
import matplotlib.pyplot as plt





# (1) 2D U-Net

def conv_2d_block(input, num_filters):
  x = Conv2D(num_filters, 3, padding="same") (input)
  #x = BatchNormalization()(x)   # no in Ronneberger et al. (2015)
  x = Activation("relu")(x)
  
  x = Conv2D(num_filters, 3, padding="same")(x)
  #x = BatchNormalization()(x)   # no in Ronneberger et al. (2015)
  x = Activation("relu")(x) 
  return x


def encoder_2d_block(input, num_filters):
  x = conv_2d_block(input, num_filters)
  p = MaxPool2D((2,2))(x)
  return x, p


def decoder_2d_block(input, skip_features, num_filters):
  x = Conv2DTranspose(num_filters, (2,2), strides=2, padding="same")(input)
  #x = UpSampling2D(num_filters, (2,2), strides=2, padding="same")(input)
  x = Concatenate()([x, skip_features])
  x = conv_2d_block(x, num_filters)
  return x


def unet_2d(input_shape, n_classes, num_filters=64, activation="sigmoid"):
  inputs = Input(input_shape)

  s1, p1 = encoder_2d_block(inputs, num_filters)
  s2, p2 = encoder_2d_block(p1, num_filters*2)
  s3, p3 = encoder_2d_block(p2, num_filters*4)
  s4, p4 = encoder_2d_block(p3, num_filters*8)

  b1 = conv_2d_block(p4, num_filters*16)

  d1 = decoder_2d_block(b1, s4, num_filters*8)
  d2 = decoder_2d_block(d1, s3, num_filters*4)
  d3 = decoder_2d_block(d2, s2, num_filters*2)
  d4 = decoder_2d_block(d3, s1, num_filters)

  outputs = Conv2D(n_classes, 1, padding="same", activation=activation)(d4)

  model = Model(inputs, outputs, name="U-Net_2D")
  model.summary()
  print("Output activation: %s" % activation)

  return model





# (2) 3D U-Net

def conv_3d_block(input, num_filters):
  x = Conv3D(num_filters, 3, padding="same") (input)
  #x = BatchNormalization()(x)   # no in Ronneberger et al. (2015)
  x = Activation("relu")(x)
  x = Conv3D(num_filters, 3, padding="same")(x)
  #x = BatchNormalization()(x)   # no in Ronneberger et al. (2015)
  x = Activation("relu")(x)
  return x


def encoder_3d_block(input, num_filters):
  x = conv_3d_block(input, num_filters)
  p = MaxPool3D((2,2,2))(x)
  return x, p


def decoder_3d_block(input, skip_features, num_filters):
  x = Conv3DTranspose(num_filters, (2,2,2), strides=2, padding="same")(input)
  #x = UpSampling3D(num_filters, (2,2,2), strides=2, padding="same")(input)
  x = Concatenate()([x, skip_features])
  x = conv_3d_block(x, num_filters)
  return x


def unet_3d(input_shape, n_classes, num_filters=64, activation="sigmoid"):
  inputs = Input(input_shape)

  s1, p1 = encoder_3d_block(inputs, num_filters)
  s2, p2 = encoder_3d_block(p1, num_filters*2)
  s3, p3 = encoder_3d_block(p2, num_filters*4)
  s4, p4 = encoder_3d_block(p3, num_filters*8)

  b1 = conv_3d_block(p4, num_filters*16)   

  d1 = decoder_3d_block(b1, s4, num_filters*8)
  d2 = decoder_3d_block(d1, s3, num_filters*4)
  d3 = decoder_3d_block(d2, s2, num_filters*2)
  d4 = decoder_3d_block(d3, s1, num_filters)

  outputs = Conv3D(n_classes, 1, padding="same", activation=activation)(d4)

  model = Model(inputs, outputs, name="U-Net_3D")
  model.summary()
  print("Output activation: %s" % activation)

  return model





# (3) TRAINING

# General training function
def train(model, X_train, y_train, batch_size, epochs, X_test, y_test, 
          save_int, accuracy, dimensions, number, show=False, shuffle=False, custom_fun=False):
    
    # - Initialization
    timer_start = datetime.now()
    
    if os.path.exists("training"):
        shutil.rmtree("training")
    os.makedirs("training")

    # - Callbacks
    csv_log = CSVLogger("training/2_loss_and_metrics.csv", separator=',', append=False)
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

