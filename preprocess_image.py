from keras.preprocessing.image import ImageDataGenerator
import numpy as np


def create_dataset(image_dir, isTest=False) : 
    # Set the seed


    seed = 42
    np.random.seed(seed)


    if isTest == False : 


        # I am goint to generate with more parameters the train dataset more than test set.
        # I am just going to generate the test test with doing just rescaling. (To generalize well the unseen data.)

        train_gen = ImageDataGenerator(rotation_range=15,
                               brightness_range=[0.4,1.5],
                               width_shift_range=0.10, 
                               height_shift_range=0.10, 
                               rescale=1/255,   
                               shear_range=0.1, 
                               zoom_range=0.1, 
                               horizontal_flip=True,
                               vertical_flip=True, 
                               fill_mode='nearest')
        train_dataset = train_gen.flow_from_directory(image_dir, 
                                              target_size=(48, 48), 
                                              color_mode='rgb', 
                                              class_mode='categorical', 
                                              batch_size=32, 
                                              shuffle=True, 
                                              seed=seed)
        
        return train_dataset
    else : 


        test_gen = ImageDataGenerator(rescale=1/255)
        test_dataset = test_gen.flow_from_directory(image_dir, 
                                              target_size=(48, 48), 
                                              color_mode='rgb', 
                                              class_mode='categorical', 
                                              batch_size=32, 
                                              shuffle=False, 
                                              seed=seed)
        return test_dataset
