#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the basic data science library

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[54]:


#importing the deep learning libraries

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow import keras
from tensorflow.keras import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.layers import BatchNormalization
from keras.optimizers import Adam
import os


# In[4]:


#fizing the image_size along with the Batch_size

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


# In[5]:


#path to he training dataset and testing dataset

train_dir = r'C:\Users\DELL\Desktop\python project\Deep Learning\Broken Egg\Dataset\train'
test_dir = r'C:\Users\DELL\Desktop\python project\Deep Learning\Broken Egg\Dataset\test'


# In[14]:


train_generator = ImageDataGenerator(
    preprocessing_function = tf.keras.applications.vgg19.preprocess_input,
    validation_split = 0.2
)
test_generator = ImageDataGenerator(
    preprocessing_function = tf.keras.applications.vgg19.preprocess_input
)


# In[16]:


#splitting the data in train catagory

train_image = train_generator.flow_from_directory(r'C:\Users\DELL\Desktop\python project\Deep Learning\Broken Egg\Dataset\train',
                                                 target_size = IMAGE_SIZE,
                                                 batch_size = BATCH_SIZE,
                                                 color_mode  = 'rgb',
                                                 class_mode = 'categorical',
                                                 shuffle = True, 
                                                 seed = 42, 
                                                 subset = 'training')


# In[18]:


#splitting the data in validation catagory

validation_image = train_generator.flow_from_directory(r'C:\Users\DELL\Desktop\python project\Deep Learning\Broken Egg\Dataset\train',
                                                       target_size = IMAGE_SIZE, 
                                                       batch_size = BATCH_SIZE,
                                                       class_mode = 'categorical',
                                                       color_mode = 'rgb', 
                                                       shuffle = True,
                                                       seed = 42,
                                                       subset = 'validation')


# In[20]:


#splitting the data in test catagory

test_image = test_generator.flow_from_directory(r'C:\Users\DELL\Desktop\python project\Deep Learning\Broken Egg\Dataset\test',
                                                target_size = IMAGE_SIZE, 
                                                batch_size = BATCH_SIZE, 
                                                class_mode = 'categorical', 
                                                color_mode = 'rgb', 
                                                shuffle = True, 
                                                seed = 42)


# In[22]:


#data visualizing in the rgb format for easy classification

labels = [k for k in train_image.class_indices]
sample_generate = train_image.__next__()

images = sample_generate[0]
titles = sample_generate[1]
plt.figure(figsize = (20 , 20))

for i in range(15):
    plt.subplot(5 , 5, i+1)
    plt.subplots_adjust(hspace = 0.3 , wspace = 0.3)
    plt.imshow(images[i])
    plt.title(f'Class: {labels[np.argmax(titles[i],axis=0)]}')
    plt.axis("off")


# In[25]:


#putting up the agument layers for VGG19

augment = tf.keras.Sequential([
  layers.experimental.preprocessing.Resizing(224,224),
  layers.experimental.preprocessing.Rescaling(1./255),
  layers.experimental.preprocessing.RandomFlip("horizontal"),
  layers.experimental.preprocessing.RandomRotation(0.1),
  layers.experimental.preprocessing.RandomZoom(0.1),
  layers.experimental.preprocessing.RandomContrast(0.1),
])


# In[27]:


#applying the model VGG19

pretrained_model = tf.keras.applications.vgg19.VGG19(
    input_shape = (224, 224, 3), 
    include_top = False, 
    weights = 'imagenet', 
    pooling = 'max'
 )
pretrained_model.trainable = False


# In[38]:


#declaring the checkpoint path

checkpoint_path = 'broken_egg_classification_checkpoint'


# In[62]:


#creating checkpoint_callback layer

checkpoint_callback = ModelCheckpoint(checkpoint_path, 
                                  save_wiights_only = True, 
                                  monitor = 'val_accuracy', 
                                  save_best_only = True)
#creating the early stopping layer
early_stopping = EarlyStopping(monitor = 'val_loss', 
                               patience = 5, 
                               restore_best_weights = True)
#creating the reduce LROn plateau layer
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', 
                              factor = 0.2, 
                              patience = 3, 
                              min_lr = 0.0001)


# In[63]:


#inputs for applying model

inputs = pretrained_model.input
x = augment(inputs)

x = Dense(128, activation = 'relu')(pretrained_model.output)
x = BatchNormalization()(x)
x = Dropout(0.45)(x)
x = Dense(256, activation = 'relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.45)(x)


# In[64]:


#output for applying model

outputs = Dense(2, activation = 'softmax')(x)


# In[65]:


#creating the model

model = Model(inputs = inputs, outputs = outputs)


# In[66]:


#compiling the model

model.compile(optimizer = Adam(0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[67]:


#creating the tensorboard_callback

from tensorflow.keras.callbacks import TensorBoard
import datetime

def create_tensorboard_callback(log_dir, experiment_name):
    log_dir = log_dir + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir)
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback


# In[68]:


#appling the model

history = model.fit(train_image, 
                    steps_per_epoch = len(train_image),
                    validation_data = validation_image, 
                    validation_steps = len(validation_image), 
                    epochs = 100, 
                    callbacks = [early_stopping,
                                 create_tensorboard_callback("training_logs","eggs_classification"),
                                 checkpoint_callback,
                                 reduce_lr])


# In[71]:


results = model.evaluate(test_image, verbose=0)

print("    Test Loss: {:.5f}".format(results[0]))
print("Test Accuracy: {:.2f}%".format(results[1] * 100))


# In[84]:


#plotting the chart for validation loss and validation accuracy against the epochs 

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(accuracy))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(epochs, accuracy, 'b', label='Training accuracy')
ax1.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
ax1.set_title('Training and validation accuracy')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Accuracy')
ax1.legend()

ax2.plot(epochs, loss, 'b', label='Training loss')
ax2.plot(epochs, val_loss, 'r', label='Validation loss')
ax2.set_title('Training and validation loss')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.legend()


# In[86]:


#saving the model

model.save('model_VGG19.h5')


# In[88]:


#predicting the model on the images

pred = model.predict(test_image)
pred = np.argmax(pred,axis=1)

# Map the label
labels = (train_image.class_indices)
labels = dict((v,k) for k,v in labels.items())
pred = [labels[k] for k in pred]

# Display the result
print(f'The first 5 predictions: {pred[:5]}')


# In[ ]:




