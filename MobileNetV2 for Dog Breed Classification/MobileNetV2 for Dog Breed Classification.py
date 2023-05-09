#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


labels_csv = pd.read_csv("labels.csv")


# In[3]:


print(labels_csv.describe())


# In[4]:


print(labels_csv.head())


# In[5]:


labels_csv.head()


# In[6]:


labels_csv["breed"].value_counts().plot.bar(figsize=(20, 10));


# In[7]:


from IPython.display import Image


# In[8]:


Image("C://Users//DELL//Desktop//python project//Machine Learning//Deep Learning//project 2//train//0021f9ceb3235effd7fcde7f7538ed62.jpg")


# In[9]:


filenames = ["C://Users//DELL//Desktop//python project//Machine Learning//Deep Learning//project 2//train//" + fname + ".jpg" for fname in labels_csv["id"]]

filenames[:10]


# In[10]:


import os
if len(os.listdir("C://Users//DELL//Desktop//python project//Machine Learning//Deep Learning//project 2//train//")) == len(filenames):
  print("proceed!")
else:
  print("error")


# In[11]:


Image(filenames[900])


# In[12]:


labels = labels_csv["breed"].to_numpy() # convert labels column to NumPy array
labels[:10]


# In[13]:


if len(labels) == len(filenames):
  print("proceed!")
else:
  print("error")


# In[14]:


unique_breeds = np.unique(labels)
len(unique_breeds)


# In[15]:


print(labels[0])
labels[0] == unique_breeds


# In[16]:


boolean_labels = [label == np.array(unique_breeds) for label in labels]
boolean_labels[:3]


# In[17]:


print(labels[0])
print(np.where(unique_breeds == labels[0])[0][0])
print(boolean_labels[0].argmax())
print(boolean_labels[0].astype(int))


# In[18]:


x = filenames
y = boolean_labels


# In[19]:


NUM_IMAGE = 1000


# In[20]:


from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x[:NUM_IMAGE],
                                                 y[:NUM_IMAGE],
                                                 test_size = 0.2,
                                                 random_state = 42)


# In[21]:


x_train[:3], y_train[:3]


# In[23]:


from matplotlib.pyplot import imread
image = imread(filenames[42])
image.shape


# In[25]:


import tensorflow as tf
import tensorflow_hub as hub


# In[26]:


tf.constant(image)[:3]


# In[27]:


IMG_SIZE = 224

def process_image(image_path):
  image = tf.io.read_file(image_path)
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
  return image


# In[28]:


def get_image_label(image_path, label):
  image = process_image(image_path)
  return image, label


# In[29]:


BATCH_SIZE = 32
def create_data_batches(x, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
  if test_data:
    print("Creating test data batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x)))
    data_batch = data.map(process_image).batch(BATCH_SIZE)
    return data_batch
  elif valid_data:
    print("Creating validation data batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x), 
                                               tf.constant(y))) 
    data_batch = data.map(get_image_label).batch(BATCH_SIZE)
    return data_batch

  else:
    print("Creating training data batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x), 
                                              tf.constant(y))) 
    data = data.shuffle(buffer_size=len(x))
    data = data.map(get_image_label)
    data_batch = data.batch(BATCH_SIZE)
  return data_batch


# In[30]:


train_data = create_data_batches(x_train, y_train)
val_data = create_data_batches(x_val, y_val, valid_data=True)


# In[31]:


train_data = create_data_batches(x_train, y_train)
val_data = create_data_batches(x_val, y_val, valid_data=True)


# In[32]:


def show_25_images(images, labels):
  plt.figure(figsize=(10, 10))
  for i in range(25):
    ax = plt.subplot(5, 5, i+1)
    plt.imshow(images[i])
    plt.title(unique_breeds[labels[i].argmax()])
    plt.axis("off")


# In[33]:


train_images, train_labels = next(train_data.as_numpy_iterator())
show_25_images(train_images, train_labels)


# In[34]:


val_images, val_labels = next(val_data.as_numpy_iterator())
show_25_images(val_images, val_labels)


# In[35]:


INPUT_SHAPE = [None, IMG_SIZE, IMG_SIZE, 3] 


# In[36]:


OUTPUT_SHAPE = len(unique_breeds)


# In[37]:


MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"


# In[38]:


def create_model(input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, model_url=MODEL_URL):
  print("Building model with:", MODEL_URL)
  model = tf.keras.Sequential([
    hub.KerasLayer(MODEL_URL), 
    tf.keras.layers.Dense(units=OUTPUT_SHAPE, 
                          activation="softmax") 
  ])
  model.compile(
      loss=tf.keras.losses.CategoricalCrossentropy(), 
      optimizer=tf.keras.optimizers.Adam(),
      metrics=["accuracy"] 
  )
  model.build(INPUT_SHAPE) 
  
  return model


# In[39]:


model = create_model()
model.summary()


# In[40]:


get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[41]:


import datetime
def create_tensorboard_callback():
  logdir = os.path.join("drive/My Drive/Data/logs",
                        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
  return tf.keras.callbacks.TensorBoard(logdir)


# In[42]:


early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                                  patience=3)


# In[43]:


NUM_EPOCHS = 100


# In[44]:


def train_model():
  model = create_model()
  tensorboard = create_tensorboard_callback()
  model.fit(x=train_data,
            epochs=NUM_EPOCHS,
            validation_data=val_data,
            validation_freq=1,
            callbacks=[tensorboard, early_stopping])
  
  return model


# In[45]:


model = train_model()


# In[47]:


from tensorflow.keras.models import load_model


# In[48]:


model.save('model_mobilenetV2.h5')


# In[58]:


predictions = model.predict(val_data, verbose=1) 
predictions


# In[59]:


predictions.shape


# In[60]:


print(predictions[0])
print(f"Max value (probability of prediction): {np.max(predictions[0])}") 
print(f"Sum: {np.sum(predictions[0])}")
print(f"Max index: {np.argmax(predictions[0])}") 
print(f"Predicted label: {unique_breeds[np.argmax(predictions[0])]}") 


# In[ ]:




