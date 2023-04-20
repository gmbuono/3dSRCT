"""
pix2pix

Isola et al. (2016): https://arxiv.org/abs/1611.07004

Further information in:
https://phillipi.github.io/pix2pix/
https://machinelearningmastery.com/how-to-implement-pix2pix-gan-models-from-scratch-with-keras/
https://machinelearningmastery.com/how-to-develop-a-pix2pix-gan-for-image-to-image-translation/
"""


# Libraries
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, Conv3D
from tensorflow.keras.layers import Conv2DTranspose, Conv3DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os, shutil 
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as sk_psnr




# (1) 2D pix2pix
    
# 1.1. Generator    

def encoder_2d_block(layer_in, n_filters, batchnorm=True):
    init = RandomNormal(stddev=0.02)
    g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    g = LeakyReLU(alpha=0.2)(g) 
    return g
 
  
  
def decoder_2d_block(layer_in, skip_in, n_filters, dropout=True):
    init = RandomNormal(stddev=0.02)
    g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    g = BatchNormalization()(g, training=True)
    if dropout:
        g = Dropout(0.5)(g, training=True)
    g = Concatenate()([g, skip_in])
    g = Activation('relu')(g)
    return g
 

 
# Define generator (a): 256
def define_generator256_2d(input_shape=(256,256,3), n_classes=3, activation='tanh'):
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=input_shape)
    
    e1 = encoder_2d_block(in_image, 64, batchnorm=False)
    e2 = encoder_2d_block(e1, 128)
    e3 = encoder_2d_block(e2, 256)
    e4 = encoder_2d_block(e3, 512)
    e5 = encoder_2d_block(e4, 512)
    e6 = encoder_2d_block(e5, 512)
    e7 = encoder_2d_block(e6, 512)
    
    b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
    b = Activation('relu')(b)
    
    d1 = decoder_2d_block(b, e7, 512)
    d2 = decoder_2d_block(d1, e6, 512)
    d3 = decoder_2d_block(d2, e5, 512)
    d4 = decoder_2d_block(d3, e4, 512, dropout=False)
    d5 = decoder_2d_block(d4, e3, 256, dropout=False)
    d6 = decoder_2d_block(d5, e2, 128, dropout=False)
    d7 = decoder_2d_block(d6, e1, 64, dropout=False)
    
    g = Conv2DTranspose(n_classes, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation(activation)(g)
    
    model = Model(in_image, out_image, name="pix2pix_2D-Generator-256")
    model.summary()
    print("Output activation for generator: %s" % activation)
    
    return model


# Define generator (b): 128
def define_generator128_2d(input_shape=(128,128,3), n_classes=3, activation='tanh'):
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=input_shape)
    
    e1 = encoder_2d_block(in_image, 64, batchnorm=False)
    e2 = encoder_2d_block(e1, 128)
    e3 = encoder_2d_block(e2, 256)
    e4 = encoder_2d_block(e3, 512)
    e5 = encoder_2d_block(e4, 512)
    e6 = encoder_2d_block(e5, 512)
    
    b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e6)
    b = Activation('relu')(b)
    
    d2 = decoder_2d_block(b, e6, 512)
    d3 = decoder_2d_block(d2, e5, 512)
    d4 = decoder_2d_block(d3, e4, 512, dropout=False)
    d5 = decoder_2d_block(d4, e3, 256, dropout=False)
    d6 = decoder_2d_block(d5, e2, 128, dropout=False)
    d7 = decoder_2d_block(d6, e1, 64, dropout=False)
    
    g = Conv2DTranspose(n_classes, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation(activation)(g)
    
    model = Model(in_image, out_image, name="pix2pix_2D-Generator-128")
    model.summary()
    print("Output activation for generator: %s" % activation)
    
    return model


# Define generator (c): 64
def define_generator64_2d(input_shape=(64,64,3), n_classes=3, activation='tanh'):
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=input_shape)
    
    e1 = encoder_2d_block(in_image, 64, batchnorm=False)
    e2 = encoder_2d_block(e1, 128)
    e3 = encoder_2d_block(e2, 256)
    e4 = encoder_2d_block(e3, 512)
    e5 = encoder_2d_block(e4, 512)
    
    b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e5)
    b = Activation('relu')(b)
    
    d3 = decoder_2d_block(b, e5, 512)
    d4 = decoder_2d_block(d3, e4, 512, dropout=False)
    d5 = decoder_2d_block(d4, e3, 256, dropout=False)
    d6 = decoder_2d_block(d5, e2, 128, dropout=False)
    d7 = decoder_2d_block(d6, e1, 64, dropout=False)
    
    g = Conv2DTranspose(n_classes, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation(activation)(g)
    
    model = Model(in_image, out_image, name="pix2pix_2D-Generator-64")
    model.summary()
    print("Output activation for generator: %s" % activation)
    
    return model


# Define generator (d): 32
def define_generator32_2d(input_shape=(32,32,3), n_classes=3, activation='tanh'):
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=input_shape)
    
    e1 = encoder_2d_block(in_image, 64, batchnorm=False)
    e2 = encoder_2d_block(e1, 128)
    e3 = encoder_2d_block(e2, 256)
    e4 = encoder_2d_block(e3, 512)
    
    b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e4)
    b = Activation('relu')(b)
    
    d4 = decoder_2d_block(b, e4, 512, dropout=False)
    d5 = decoder_2d_block(d4, e3, 256, dropout=False)
    d6 = decoder_2d_block(d5, e2, 128, dropout=False)
    d7 = decoder_2d_block(d6, e1, 64, dropout=False)
    
    g = Conv2DTranspose(n_classes, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation(activation)(g)
    
    model = Model(in_image, out_image, name="pix2pix_2D-Generator-64")
    model.summary()
    print("Output activation for generator: %s" % activation)
    
    return model




# 1.2. Discriminator
def define_discriminator_2d(input_shape):
    init = RandomNormal(stddev=0.02)
    in_src_image = Input(shape=input_shape)
    in_target_image = Input(shape=input_shape)
    merged = Concatenate()([in_src_image, in_target_image])
    
    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)

    model = Model([in_src_image, in_target_image], patch_out, name="pix2pix_2D-Discriminator")
    model.summary()

    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    
    return model
 
    
 
 
# 1.3. GAN model 
def define_gan_2d(g_model, d_model, input_shape):
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    
    in_src = Input(shape=input_shape)
    
    gen_out = g_model(in_src)
    dis_out = d_model([in_src, gen_out])

    model = Model(in_src, [dis_out, gen_out], name="pix2pix_2D-GAN")
    model.summary()

    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
    
    return model





# (2) 3D pix2pix
    
# 1.1. Generator    

def encoder_3d_block(layer_in, n_filters, batchnorm=True):
    init = RandomNormal(stddev=0.02)
    g = Conv3D(n_filters, 4, strides=2, padding='same', kernel_initializer=init)(layer_in)
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    g = LeakyReLU(alpha=0.2)(g) 
    return g
 
 
   
def decoder_3d_block(layer_in, skip_in, n_filters, dropout=True):
    init = RandomNormal(stddev=0.02)
    g = Conv3DTranspose(n_filters, 4, strides=2, padding='same', kernel_initializer=init)(layer_in)
    g = BatchNormalization()(g, training=True)
    if dropout:
        g = Dropout(0.5)(g, training=True)
    g = Concatenate()([g, skip_in])
    g = Activation('relu')(g)
    return g
 


# Define generator (a): 256
def define_generator256_3d(input_shape=(256,256, 256,3), n_classes=3, activation='tanh'):
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=input_shape)
    
    e1 = encoder_3d_block(in_image, 64, batchnorm=False)
    e2 = encoder_3d_block(e1, 128)
    e3 = encoder_3d_block(e2, 256)
    e4 = encoder_3d_block(e3, 512)
    e5 = encoder_3d_block(e4, 512)
    e6 = encoder_3d_block(e5, 512)
    e7 = encoder_3d_block(e6, 512)
    
    b = Conv3D(512, 4, strides=2, padding='same', kernel_initializer=init)(e7)
    b = Activation('relu')(b)
    
    d1 = decoder_3d_block(b, e7, 512)
    d2 = decoder_3d_block(d1, e6, 512)
    d3 = decoder_3d_block(d2, e5, 512)
    d4 = decoder_3d_block(d3, e4, 512, dropout=False)
    d5 = decoder_3d_block(d4, e3, 256, dropout=False)
    d6 = decoder_3d_block(d5, e2, 128, dropout=False)
    d7 = decoder_3d_block(d6, e1, 64, dropout=False)
    
    g = Conv3DTranspose(n_classes, 4, strides=2, padding='same', kernel_initializer=init)(d7)
    out_image = Activation(activation)(g)
    
    model = Model(in_image, out_image, name="pix2pix_3D-Generator-256")
    model.summary()
    print("Output activation for generator: %s" % activation)
    
    return model


# Define generator (b): 128
def define_generator128_3d(input_shape=(128,128,128,3), n_classes=3, activation='tanh'):
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=input_shape)
    
    e1 = encoder_3d_block(in_image, 64, batchnorm=False)
    e2 = encoder_3d_block(e1, 128)
    e3 = encoder_3d_block(e2, 256)
    e4 = encoder_3d_block(e3, 512)
    e5 = encoder_3d_block(e4, 512)
    e6 = encoder_3d_block(e5, 512)
    
    b = Conv3D(512, 4, strides=2, padding='same', kernel_initializer=init)(e6)
    b = Activation('relu')(b)
    
    d2 = decoder_3d_block(b, e6, 512)
    d3 = decoder_3d_block(d2, e5, 512)
    d4 = decoder_3d_block(d3, e4, 512, dropout=False)
    d5 = decoder_3d_block(d4, e3, 256, dropout=False)
    d6 = decoder_3d_block(d5, e2, 128, dropout=False)
    d7 = decoder_3d_block(d6, e1, 64, dropout=False)
    
    g = Conv3DTranspose(n_classes, 4, strides=2, padding='same', kernel_initializer=init)(d7)
    out_image = Activation(activation)(g)
    
    model = Model(in_image, out_image, name="pix2pix_3D-Generator-128")
    model.summary()
    print("Output activation for generator: %s" % activation)
    
    return model


# Define generator (c): 64
def define_generator64_3d(input_shape=(64,64,64,3), n_classes=3, activation='tanh'):
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=input_shape)
    
    e1 = encoder_3d_block(in_image, 64, batchnorm=False)
    e2 = encoder_3d_block(e1, 128)
    e3 = encoder_3d_block(e2, 256)
    e4 = encoder_3d_block(e3, 512)
    e5 = encoder_3d_block(e4, 512)
    
    b = Conv3D(512, 4, strides=2, padding='same', kernel_initializer=init)(e5)
    b = Activation('relu')(b)
    
    d3 = decoder_3d_block(b, e5, 512)
    d4 = decoder_3d_block(d3, e4, 512, dropout=False)
    d5 = decoder_3d_block(d4, e3, 256, dropout=False)
    d6 = decoder_3d_block(d5, e2, 128, dropout=False)
    d7 = decoder_3d_block(d6, e1, 64, dropout=False)
    
    g = Conv3DTranspose(n_classes, 4, strides=2, padding='same', kernel_initializer=init)(d7)
    out_image = Activation(activation)(g)
    
    model = Model(in_image, out_image, name="pix2pix_3D-Generator-64")
    model.summary()
    print("Output activation for generator: %s" % activation)
    
    return model


# Define generator (d): 32
def define_generator32_3d(input_shape=(32,32,32,3), n_classes=3, activation='tanh'):
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=input_shape)
    
    e1 = encoder_3d_block(in_image, 64, batchnorm=False)
    e2 = encoder_3d_block(e1, 128)
    e3 = encoder_3d_block(e2, 256)
    e4 = encoder_3d_block(e3, 512)
    
    b = Conv3D(512, 4, strides=2, padding='same', kernel_initializer=init)(e4)
    b = Activation('relu')(b)
    
    d4 = decoder_3d_block(b, e4, 512, dropout=False)
    d5 = decoder_3d_block(d4, e3, 256, dropout=False)
    d6 = decoder_3d_block(d5, e2, 128, dropout=False)
    d7 = decoder_3d_block(d6, e1, 64, dropout=False)
    
    g = Conv3DTranspose(n_classes, 4, strides=2, padding='same', kernel_initializer=init)(d7)
    out_image = Activation(activation)(g)
    
    model = Model(in_image, out_image, name="pix2pix_3D-Generator-64")
    model.summary()
    print("Output activation for generator: %s" % activation)
    
    return model




# 1.2. Discriminator 
def define_discriminator_3d(input_shape):
    init = RandomNormal(stddev=0.02)
    in_src_image = Input(shape=input_shape)
    in_target_image = Input(shape=input_shape)
    merged = Concatenate()([in_src_image, in_target_image])
    
    d = Conv3D(64, 4, strides=2, padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = Conv3D(128, 4, strides=2, padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = Conv3D(256, 4, strides=2, padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = Conv3D(512, 4, strides=2, padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = Conv3D(512, 4, padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    
    d = Conv3D(1, 4, padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)

    model = Model([in_src_image, in_target_image], patch_out, name="pix2pix_3D-Discriminator")
    model.summary()

    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    
    return model
 
    
 

# 1.3. GAN model 
def define_gan_3d(g_model, d_model, input_shape):
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    
    in_src = Input(shape=input_shape)
    
    gen_out = g_model(in_src)
    dis_out = d_model([in_src, gen_out])

    model = Model(in_src, [dis_out, gen_out], name="pix2pix_3D-GAN")
    model.summary()

    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
    
    return model





# (3) Training 

# 3.1. Generate real samples
def generate_real_samples(dataset, n_samples, patch_shape, dimensions):
    trainA, trainB = dataset
    ix = np.random.randint(0, trainA.shape[0], n_samples)
    X1, X2 = trainA[ix], trainB[ix]
    if dimensions=="2d":
        y = np.ones((n_samples, patch_shape, patch_shape, 1))
    if dimensions=="3d":
        y = np.ones((n_samples, patch_shape, patch_shape, patch_shape, 1))
    return [X1, X2], y




# 3.2. Generate fake samples
def generate_fake_samples(g_model, samples, patch_shape, dimensions):
    X = g_model.predict(samples)
    if dimensions=="2d":
        y = np.zeros((len(X), patch_shape, patch_shape, 1))
    if dimensions=="3d":
        y = np.zeros((len(X), patch_shape, patch_shape, patch_shape, 1))
    return X, y




# 3.3. Training
def train(d_model, g_model, gan_model, dataset, dataset_val, epochs=100, 
          n_batch=1, n_patch=16, save_int=5, dimensions="2d", number=False, show=False):
    
    # - Data unpacking, calculation of number of batches per training epoch and training iterations
    trainA, trainB = dataset
    bat_per_epo = int(len(trainA) / n_batch)
    n_steps = bat_per_epo * epochs
    
    
    # - Training and "callbacks"
    timer_start = datetime.now()
    
    if os.path.exists("training"):
        shutil.rmtree("training")
    os.makedirs("training")
    
    with open("training/2_loss_and_metrics.csv", 'w') as f:             
        
        for i in range(n_steps):
            # Select a batch of real and fake samples
            [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch, dimensions)
            X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch, dimensions)
            
            # Update discriminator for real and fake samples, and the generator
            d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
            d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
            g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])        
            
            # Print loss for each step, save loss and metrics each epoch 
            line = '%d, d1, d2, g, %f, %f, %f' % (i, d_loss1, d_loss2, g_loss)
            print(line)
            
            # Save loss and metrics on a random batch
            if i in np.arange (0, n_steps+1, bat_per_epo):
                [X_test_realA, X_test_realB], _ = generate_real_samples(dataset_val, n_batch, n_patch, dimensions)
                X_test_fakeB, _ = generate_fake_samples(g_model, X_test_realA, 1, dimensions)
                psnr = sk_psnr(X_test_realB, X_test_fakeB)
                line_2_write = '%d, d1, d2, g, psnr, %f, %f, %f, %f' % (i, d_loss1, d_loss2, g_loss, psnr)
                print(line_2_write)
                f.write(line_2_write)
                f.write('\n')
            
            # Summarize model performance each "SAVE_INT" epochs
            if save_int!=epochs:
                if i in np.arange ((bat_per_epo*save_int)-1, n_steps+1, bat_per_epo*save_int):
                    # Plot patches
                    from utils import pred_on_patch
                    fig,_ = pred_on_patch(g_model, dataset_val[0], dataset_val[1], dimensions, number, show)
                    fig.savefig("training/pred_patch_step%d.tif" % i)
                    # Save generator model
                    g_model.save("training/model_step%d.hdf5" % i)
                    g_model.save_weights("training/model_weights_step%d.hdf5" % i)
                    print("Saved: Plot and model at step %d" % i)
            
                    
    # - Saving
    # Save (1): model
    g_model.save("training/1_model.hdf5") 
    g_model.save_weights("training/1_model_weights.hdf5")    
      
    # Save (2): loss and metric plots
    data = pd.read_csv("training/2_loss_and_metrics.csv", header=None)
    fig1 = plt.figure(figsize=(10,5))
    plt.subplot(121).plot(data[0], data[5], label="d1_loss")
    plt.subplot(121).plot(data[0], data[6], label="d2_loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss on a batch")  
    plt.legend()
    plt.subplot(122).plot(data[0], data[7], c="g" , label="g_loss")
    plt.xlabel("Steps") 
    plt.legend()
    plt.show()
    fig1.savefig("training/2_loss_plot.tif")
    
    fig2 = plt.figure(figsize=(7,5))
    plt.plot(data[0], data[8])
    plt.title("PSNR on batch")
    plt.xlabel("Steps")
    plt.ylabel("Accuracy")
    plt.show()
    fig2.savefig("training/2_metric_plot.tif")
   
    # Save (3): training time    
    timer_end = datetime.now()
    execution_time = timer_end-timer_start
    with open("training/3_training_time.txt", 'w') as f:
        f.write(str(execution_time))
    
    return print("Training completed in: ", execution_time)

