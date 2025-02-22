import cv2
import tensorflow as tf
import numpy as np

def preprocess_frame(frame): 

    
    """
    This output is for the model's predictions. Not for showing the image.
    If you want to show the preprocess_frame's frame, you have to take the frame out of the batch with using np.array.
    Like that (np.array(frame)[0]),
    Since you do not need batches for
    showing the image. And convert GRAY to BGR to better results.
    """
    
    frame = cv2.resize(frame, (48, 48))

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    

    frame = frame / 255.0

    frame = tf.expand_dims(frame, axis=0)

    return frame
    