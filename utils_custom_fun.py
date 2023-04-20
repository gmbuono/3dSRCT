# Libraries
import numpy as np





# CUSTOM FUNCTIONS

# - Costomized learning rate
def custom1_lr(callbacks_list):
    learning_rate = 0.001

    # Linear decay
    #def decay(epoch):
        #decay_constant = -0.000025
        #lrate = learning_rate + (decay_constant*epoch)
        #return lrate

    # Exponential decay
    def decay(epoch):
        #decay_rate = learning_rate / epochs   
        decay_rate = 0.0625
        lrate = learning_rate * np.exp(-decay_rate*epoch)
        return lrate

    from tensorflow.keras.callbacks import LearningRateScheduler
    lr_function = LearningRateScheduler(decay, verbose=1)

    callbacks_list.append(lr_function)

