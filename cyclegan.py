"""
CycleGAN 

Zhu et al. (2017): https://arxiv.org/abs/1703.10593

Further information in:
https://junyanz.github.io/CycleGAN/
https://machinelearningmastery.com/how-to-develop-cyclegan-models-from-scratch-with-keras/
https://machinelearningmastery.com/cyclegan-tutorial-with-keras/
"""


# Libraries
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, Conv3D
from tensorflow.keras.layers import Conv2DTranspose, Conv3DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers import InstanceNormalization
from numpy import zeros, ones, asarray, arange
from numpy.random import randint
from random import random
from datetime import datetime
import os, shutil 
from matplotlib import pyplot as plt
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as sk_psnr




# (1) 2D CycleGAN

# 1.1. Generator (based on ResNet)

def resnet_block_2d(n_filters, input_layer):
	init = RandomNormal(stddev=0.02)
	g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	g = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Concatenate()([g, input_layer])
	return g

    
    
def define_generator_cyclegan_2d(image_shape, n_classes, n_resnet, activation='tanh'):
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=image_shape)
    g = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(in_image)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    g = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    g = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    for _ in range(n_resnet):
        g = resnet_block_2d(256, g)
    g = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    g = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    g = Conv2D(n_classes, (7,7), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    out_image = Activation(activation)(g)
    model = Model(in_image, out_image)
    model.summary() 
    print("Output activation for generator: %s" % activation)
    return model
 
    
 
    
# 1.2. Discriminator 
def define_discriminator_2d(image_shape):
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=image_shape)
    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    patch_out = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
    model = Model(in_image, patch_out)
    model.summary()
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss_weights=[0.5]) 
    return model
   


  
# 1.3 Composite model for updating generators by adversarial and cycle loss
def define_composite_model_2d(g_model_1, d_model, g_model_2, image_shape):
    g_model_1.trainable = True
    d_model.trainable = False
    g_model_2.trainable = False
	# discriminator element
    input_gen = Input(shape=image_shape)
    gen1_out = g_model_1(input_gen)
    output_d = d_model(gen1_out)
	# identity element
    input_id = Input(shape=image_shape)
    output_id = g_model_1(input_id)
	# forward cycle
    output_f = g_model_2(gen1_out)
	# backward cycle
    gen2_out = g_model_2(input_id)
    output_b = g_model_1(gen2_out)

    model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt) 
    model.summary() 
    return model
 
 
    

   
# (2) 3D CycleGAN

# 2.1. Generator (based on ResNet)
   
def resnet_block_3d(n_filters, input_layer):
	init = RandomNormal(stddev=0.02)
	g = Conv3D(n_filters, 3, padding='same', kernel_initializer=init)(input_layer)
	g = InstanceNormalization(axis=-1)(g)
	g = Activation('relu')(g)
	g = Conv3D(n_filters, 3, padding='same', kernel_initializer=init)(g)
	g = InstanceNormalization(axis=-1)(g)
	g = Concatenate()([g, input_layer])
	return g
 
   
 
def define_generator_cyclegan_3d(image_shape, n_classes, n_resnet, activation='tanh'):
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=image_shape)
    g = Conv3D(64, 7, padding='same', kernel_initializer=init)(in_image)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    g = Conv3D(128, 3, strides=2, padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    g = Conv3D(256, 3, strides=2, padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    for _ in range(n_resnet):
        g = resnet_block_3d(256, g)
    g = Conv3DTranspose(128, 3, strides=2, padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    g = Conv3DTranspose(64, 3, strides=2, padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    g = Conv3D(n_classes, 7, padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    out_image = Activation(activation)(g)
    model = Model(in_image, out_image)
    model.summary() 
    print("Output activation for generator: %s" % activation)
    return model




# 2.2. Discriminator 
def define_discriminator_3d(image_shape):
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=image_shape)
    d = Conv3D(64, 4, strides=2, padding='same', kernel_initializer=init)(in_image)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv3D(128, 4, strides=2, padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv3D(256, 4, strides=2, padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv3D(512, 4, strides=2, padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv3D(512, 4, padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)
    patch_out = Conv3D(1, 4, padding='same', kernel_initializer=init)(d)
    model = Model(in_image, patch_out)
    model.summary()
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), loss_weights=[0.5]) 
    return model    
 
   

    
# 2.3 Composite model for updating generators by adversarial and cycle loss
def define_composite_model_3d(g_model_1, d_model, g_model_2, image_shape):
    g_model_1.trainable = True
    d_model.trainable = False
    g_model_2.trainable = False
	# discriminator element
    input_gen = Input(shape=image_shape)
    gen1_out = g_model_1(input_gen)
    output_d = d_model(gen1_out)
	# identity element
    input_id = Input(shape=image_shape)
    output_id = g_model_1(input_id)
	# forward cycle
    output_f = g_model_2(gen1_out)
	# backward cycle
    gen2_out = g_model_2(input_id)
    output_b = g_model_1(gen2_out)

    model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt) 
    model.summary() 
    return model    
 
    
    
 
    
# (3) Training 
# 3.1. Generate real samples
def generate_real_samples(dataset, n_samples, patch_shape, dimensions):
    ix = randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    if dimensions=="2d":
        y = ones((n_samples, patch_shape, patch_shape, 1))
    if dimensions=="3d":
        y = ones((n_samples, patch_shape, patch_shape, patch_shape, 1)) 
    return X, y
 
    
 
    
# 3.2. Generate fake samples
def generate_fake_samples(g_model, dataset, patch_shape, dimensions):
    X = g_model.predict(dataset)
    if dimensions=="2d":
        y = zeros((len(X), patch_shape, patch_shape, 1))
    if dimensions=="3d":
        y = zeros((len(X), patch_shape, patch_shape, patch_shape, 1)) 
    return X, y
 
    
 
    
# 3.3. Update image pool for fake images
def update_image_pool(pool, images, max_size=50):
	selected = list()
	for image in images:
		if len(pool) < max_size:
			# stock the pool
			pool.append(image)
			selected.append(image)
		elif random() < 0.5:
			# use image, but don't add it to the pool
			selected.append(image)
		else:
			# replace an existing image and use replaced image
			ix = randint(0, len(pool))
			selected.append(pool[ix])
			pool[ix] = image
	return asarray(selected) 
   

    
     
#3.4. Training
def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, 
          dataset, dataset_val, epochs=25, dimensions="2d", save_int=5, number=False, show=False):
    
    # Define properties of the training run and prepare data
    n_epochs, n_batch, = epochs, 1
    n_patch = d_model_A.output_shape[1] 
    
    trainA, trainB = dataset 
    trainA_test, trainB_test = dataset_val 
    poolA, poolB = list(), list() 
    
    bat_per_epo = int(len(trainA) / n_batch) 
    n_steps = bat_per_epo * n_epochs
    
     
    # - Training and "callbacks"
    timer_start = datetime.now()
    
    if os.path.exists("training"):
        shutil.rmtree("training")
    os.makedirs("training")
    
    with open("training/2_loss_and_metrics.csv", 'w') as f:             
        
        for i in range(n_steps):
            # select a batch of real samples
            X_realA, y_realA = generate_real_samples(trainA, n_batch, n_patch, dimensions) 
            X_realB, y_realB = generate_real_samples(trainB, n_batch, n_patch, dimensions) 
            # generate a batch of fake samples 
            X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB, n_patch, dimensions) 
            X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA, n_patch, dimensions) 
            # update fakes from pool 
            X_fakeA = update_image_pool(poolA, X_fakeA) 
            X_fakeB = update_image_pool(poolB, X_fakeB) 
            
            # update generator B->A via adversarial and cycle loss 
            g_loss2, _, _, _, _  = c_model_BtoA.train_on_batch([X_realB, X_realA], 
                                                               [y_realA, X_realA, X_realB, X_realA]) 
            # update discriminator for A -> [real/fake] 
            dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA) 
            dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA) 
            # update generator A->B via adversarial and cycle loss 
            g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], 
                                                              [y_realB, X_realB, X_realA, X_realB]) 
            # update discriminator for B -> [real/fake] 
            dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB) 
            dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)
		
        
            # Print loss for each step, save loss and metrics each epoch 
            #print('>%d, dA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (i+1, dA_loss1,dA_loss2, dB_loss1,dB_loss2, g_loss1,g_loss2)) 
            line = '%d, dA1, dA2, dB1, dB2, g1, g2, %f, %f, %f, %f, %f, %f' % (i, dA_loss1,dA_loss2, dB_loss1,dB_loss2, g_loss1,g_loss2)            
            print(line)
            
                        
            # Save loss and metrics on a random batch
            if i in arange (0, n_steps+1, bat_per_epo):
                ix = randint(0, trainA_test.shape[0], n_batch)
                X_test_realA, X_test_realB = trainA_test[ix], trainB_test[ix]
                X_test_fakeA = g_model_BtoA.predict(X_test_realB) 
                X_test_fakeB = g_model_AtoB.predict(X_test_realA)
                psnr_AtoB = sk_psnr(X_test_realB, X_test_fakeB)
                psnr_BtoA = sk_psnr(X_test_realA, X_test_fakeA)
                
                line_2_write = '%d, dA1, dA2, dB1, dB2, g1, g2, psnr_AtoB, psnr_BtoA, %f, %f, %f, %f, %f, %f, %f, %f' % (i, dA_loss1,dA_loss2, dB_loss1,dB_loss2, g_loss1,g_loss2, psnr_AtoB, psnr_BtoA)
                print(line_2_write)
                f.write(line_2_write)
                f.write('\n')
            
            
            # Summarize model performance each "SAVE_INT" epochs
            if save_int!=epochs:
                if i in arange ((bat_per_epo*save_int)-1, n_steps+1, bat_per_epo*save_int):
                    # Plot patches
                    from utils import pred_on_patch
                    fig,_ = pred_on_patch(g_model_AtoB, dataset_val[0], dataset_val[1], dimensions, number, show)
                    fig.savefig("training/pred_patch_AtoB_step%d.tif" % i)
                    fig,_ = pred_on_patch(g_model_BtoA, dataset_val[1], dataset_val[0], dimensions, number, show)
                    fig.savefig("training/pred_patch_BtoA_step%d.tif" % i)
                    # Save generator model
                    g_model_AtoB.save("training/g_model_AtoB_step%d.hdf5" % i)
                    g_model_AtoB.save_weights("training/g_model_AtoB_weights_step%d.hdf5" % i)
                    g_model_BtoA.save("training/g_model_BtoA_step%d.hdf5" % i)
                    g_model_BtoA.save_weights("training/g_model_BtoA_weights_step%d.hdf5" % i)
                    print("Saved: Plot and model at step %d" % i)

    
    # - Saving
    # Save (1): model    	
    g_model_AtoB.save("training/1_g_model_AtoB.hdf5")
    g_model_AtoB.save_weights("training/1_g_model_AtoB_weights.hdf5")
    g_model_BtoA.save("training/1_g_model_BtoA.hdf5")
    g_model_BtoA.save_weights("training/1_g_model_BtoA_weights.hdf5")
    
	  
    # Save (2): loss and metric plots
    data = pd.read_csv("training/2_loss_and_metrics.csv", header=None)
    
    fig1 = plt.figure(figsize=(10,15))
    plt.subplot(321).plot(data[0], data[9], label="dA1")
    plt.subplot(321).set_xlabel("Steps")
    plt.subplot(321).set_ylabel("Loss on a batch")
    plt.subplot(321).legend()
    plt.subplot(322).plot(data[0], data[10], label="dA2")
    plt.subplot(322).set_xlabel("Steps")
    plt.subplot(322).set_ylabel("Loss on a batch")
    plt.subplot(322).legend()
    plt.subplot(323).plot(data[0], data[11], label="dB1")
    plt.subplot(323).set_xlabel("Steps")
    plt.subplot(323).set_ylabel("Loss on a batch")
    plt.subplot(323).legend()
    plt.subplot(324).plot(data[0], data[12], label="dB2")
    plt.subplot(324).set_xlabel("Steps")
    plt.subplot(324).set_ylabel("Loss on a batch")
    plt.subplot(324).legend()
    plt.subplot(325).plot(data[0], data[13], label="g1")
    plt.subplot(325).set_xlabel("Steps")
    plt.subplot(325).set_ylabel("Loss on a batch")
    plt.subplot(325).legend()
    plt.subplot(326).plot(data[0], data[14], label="g2")
    plt.subplot(326).set_xlabel("Steps")
    plt.subplot(326).set_ylabel("Loss on a batch")
    plt.subplot(326).legend()
    plt.show() 
    fig1.savefig("training/2_loss_plot.tif")
    
    fig2 = plt.figure(figsize=(10,5))
    plt.subplot(121).plot(data[0], data[15], label="psnr_AtoB")
    plt.subplot(121).set_xlabel("Steps")
    plt.subplot(121).set_ylabel("Accuracy")
    plt.subplot(121).legend()
    plt.subplot(122).plot(data[0], data[16], label="psnr_BtoA")
    plt.subplot(122).set_xlabel("Steps")
    plt.subplot(122).set_ylabel("Accuracy")
    plt.subplot(122).legend()
    plt.show()
    fig2.savefig("training/2_metric_plot.tif")
    
    
    # Save (3): training time    
    timer_end = datetime.now()
    execution_time = timer_end-timer_start
    with open("training/3_training_time.txt", 'w') as f:
        f.write(str(execution_time))
    
    return print("Training completed in: ", execution_time)
