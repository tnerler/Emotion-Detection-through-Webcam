from model import create_model
import numpy as np
import keras
import tensorflow
from preprocess_image import create_dataset


from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

base_model = create_model()

# Let's prepare our datasets.
train_path = "Emotions/train"
test_path = "Emotions/test"

train_dataset = create_dataset(train_path, isTest=False)
test_dataset = create_dataset(test_path, isTest=True)

# Compile the model
base_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Create callback functions

log_dir = "Emotions Detection/log"

callbacks = [
    ModelCheckpoint("best_model_1.h5", monitor='val_loss', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1),
    TensorBoard(log_dir=log_dir)
]

history = base_model.fit(train_dataset, epochs=100, callbacks=callbacks, validation_data=test_dataset)


