#!/usr/bin/env python
# coding: utf-8

# In[14]:


from tensorflow.keras.preprocessing import image


# In[17]:


model = load_model('model_inceptionV3.h5')


# In[53]:


img = image.load_img(r'C:\Users\DELL\Desktop\python project\Deep Learning\Potato Leaf Disease\Dataset\Training\Early_Blight\Early_Blight_302.jpg',
target_size = (224, 224))


# In[54]:


img


# In[55]:


x = image.img_to_array(img)


# In[56]:


x = x/ 225


# In[57]:


from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image


# In[58]:


x = np.expand_dims(x, axis = 0)
img_data = preprocess_input(x)
img_data.shape


# In[59]:


preds = model.predict(x)
preds = np.argmax(preds, axis =1)
if preds == 1:
    preds = 'About To Rot'
elif preds == 2:
    preds = 'Fine Leaf'
else:
    preds == 'Rotten Leaf'
print(preds)


# In[ ]:





# In[ ]:




