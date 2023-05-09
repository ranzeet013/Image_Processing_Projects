#!/usr/bin/env python
# coding: utf-8

# In[74]:


#import python libraries

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# In[75]:


import warnings 
warnings.filterwarnings('ignore')


# In[76]:


#deep learning libaries

from tensorflow import keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.utils import plot_model
from glob import glob
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Softmax
from tensorflow.keras.optimizers import Adam


# In[77]:


#image size and data paths

IMAGE_SIZE = [224, 224]
valid_dir = r'C:\Users\DELL\Desktop\python project\Deep Learning\Car Brand Detector\dataset\Images\Test'
train_dir = r'C:\Users\DELL\Desktop\python project\Deep Learning\Car Brand Detector\dataset\Images\Train'


# In[78]:


#building the model
resnet = ResNet50(include_top = False, input_shape = IMAGE_SIZE + [3], weights = 'imagenet')


# In[79]:


resnet.summary()


# In[80]:


for layer in resnet.layers:
    layer.trainable = False


# In[81]:


folder = glob(r'C:\Users\DELL\Desktop\python project\Deep Learning\Car Brand Detector\dataset\Images\Train\*')


# In[82]:


folder


# In[83]:


#adding the layers

x = Flatten()(resnet.output)


# In[84]:


#creating output layer

prediction = Dense(len(folder), activation = 'softmax')(x)


# In[85]:


#creating the model

model = Model(inputs = resnet.input, outputs = prediction)


# In[86]:


model


# In[87]:


model.summary()


# In[88]:


#compiling the model

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[89]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[90]:


train_data_generator = ImageDataGenerator(rescale = 1./ 255,
                                          shear_range = 0.2, 
                                          zoom_range = 0.2, 
                                          horizontal_flip = True)

test_data_generator = ImageDataGenerator(rescale = 1./ 255)


# In[91]:


training_set = train_data_generator.flow_from_directory( r'C:\Users\DELL\Desktop\python project\Deep Learning\Car Brand Detector\dataset\Images\Train',
                                                        target_size = (224, 224), batch_size = 32, class_mode = 'categorical')


# In[92]:


test_set = test_data_generator.flow_from_directory( r'C:\Users\DELL\Desktop\python project\Deep Learning\Car Brand Detector\dataset\Images\Test',
                                                   target_size = (224, 224), batch_size = 32, class_mode = 'categorical')


# In[93]:


#applying the model

r = model.fit_generator(training_set, 
                        validation_data = test_set,
                        epochs = 50,
                        steps_per_epoch = len(training_set),
                        validation_steps = len(test_set))


# In[94]:


plt.plot(r.history['loss'], label = 'train loss')
plt.plot(r.history['val_loss'], label = 'val_loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

#plotting the accuracy 

plt.plot(r.history['accuracy'], label = 'train acc')
plt.plot(r.history['val_accuracy'], label = 'val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


# In[95]:


from tensorflow.keras.models import load_model

model.save('model_resnet50.h5')


# In[96]:


y_pred = model.predict(test_set)


# In[97]:


y_pred


# In[98]:


#y predictions

y_pred = np.argmax(y_pred, axis = 1)


# In[99]:


from tensorflow.keras.preprocessing import image


# In[100]:


model = load_model('model_resnet50.h5')


# In[101]:


#loading the image

img = image.load_img(r'C:\Users\DELL\Desktop\python project\Deep Learning\Car Brand Detector\dataset\Images\Test\lamborghini\23.jpg', 
                     target_size = (224, 224))


# In[102]:


img


# In[103]:


x = image.img_to_array(img)


# In[104]:


x = x/ 225


# In[105]:


from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image


# In[106]:


#cleaning the image to predict 

x = np.expand_dims(x, axis = 0)
img_data = preprocess_input(x)
img_data.shape


# In[107]:


#predicting weather the car is Audi, Lamborgini or Mercedes

preds = model.predict(x)
preds = np.argmax(preds, axis =1)
if preds == 1:
    preds = 'The Car Is Audi'
elif preds == 2:
    preds = 'The Car Is Lamborgini'
else:
    preds == 'The Car Is Mercedes'
print(preds)    


# In[ ]:




