#!/usr/bin/env python
# coding: utf-8

# In[48]:


#importing the data science library

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


get_ipython().run_line_magic('matplotlib', 'inline')


# In[49]:


#importing the deep learning libraries

from tensorflow import keras
from keras.applications.inception_v3 import InceptionV3
from glob import glob
from keras.layers import Dense, Flatten
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import img_to_array, load_img, array_to_img


# In[50]:


#declearing the image size and providing the data path

IMAGE_SIZE = [224, 224]
train_dir = r'C:\Users\DELL\Desktop\python project\Deep Learning\Potato Leaf Disease\Dataset\Training'
valid_dir = r'C:\Users\DELL\Desktop\python project\Deep Learning\Potato Leaf Disease\Dataset\Validation'


# In[51]:


#making the inception laye

inception = InceptionV3(include_top = False, weights = 'imagenet', input_shape = IMAGE_SIZE + [3])


# In[52]:


inception


# In[53]:


for layer in inception.layers:
    layer.trainable = False


# In[54]:


#will be used as output layes 

folder = glob(r'C:\Users\DELL\Desktop\python project\Deep Learning\Potato Leaf Disease\Dataset\Training\*')


# In[55]:


folder


# In[56]:


#flattening the output of inception model

x = Flatten()(inception.output)


# In[57]:


x


# In[58]:


#predicting

prediction = Dense(len(folder), activation = 'softmax')(x)


# In[59]:


prediction


# In[60]:


#creating the model

model = Model(inputs = inception.input, outputs = prediction)


# In[61]:


model


# In[62]:


model.summary()


# In[72]:


#compiling the model

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[73]:


#generating the train iamge data 

train_data_generator = ImageDataGenerator(rescale = 1./225, 
                                          shear_range = 0.2, 
                                          horizontal_flip = True,
                                          zoom_range = 0.2)
#generating the test image data

test_data_generator = ImageDataGenerator(rescale = 1./225)


# In[74]:


training_set = train_data_generator.flow_from_directory(r'C:\Users\DELL\Desktop\python project\Deep Learning\Potato Leaf Disease\Dataset\Training',
                                                        target_size = (224, 224),
                                                        batch_size = 32,
                                                        class_mode = 'categorical')


# In[75]:


testing_set = test_data_generator.flow_from_directory(r'C:\Users\DELL\Desktop\python project\Deep Learning\Potato Leaf Disease\Dataset\Validation',
                                                      target_size = (224, 224),
                                                      batch_size = 32,
                                                      class_mode = 'categorical')


# In[79]:


#visualizing the images along with their classification

labels = [k for k in training_set.class_indices]
sample_generate = training_set.__next__()
images = sample_generate[0]
titles = sample_generate[1]
plt.figure(figsize = (20 , 20))

for i in range(15):
    plt.subplot(5 , 5, i+1)
    plt.subplots_adjust(hspace = 0.3 , wspace = 0.3)
    plt.imshow(images[i])
    plt.title(f'Class: {labels[np.argmax(titles[i],axis=0)]}')
    plt.axis("off")


# In[76]:


early = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5)


# In[77]:


history = model.fit_generator(
    training_set,
    validation_data=testing_set,
    epochs=30,
    steps_per_epoch=len(training_set),
    validation_steps=len(testing_set),
    callbacks=[early]
)


# In[80]:


from tensorflow.keras.models import load_model
model.save('model_inceptionV3.h5')


# In[81]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)


# In[82]:


# Accuracy plot
plt.plot(epochs, acc, color='green', label='Training Accuracy')
plt.plot(epochs, val_acc, color='blue', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.figure()


# In[83]:


# Loss plot
plt.plot(epochs, loss, color='pink', label='Training Loss')
plt.plot(epochs, val_loss, color='red', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[84]:


#visualizing the validation loo and validation accuracy graph

plt.plot(history.history['val_loss'], label = 'training loss')
plt.plot(history.history['val_accuracy'], label = 'training accuracy')
plt.legend()

