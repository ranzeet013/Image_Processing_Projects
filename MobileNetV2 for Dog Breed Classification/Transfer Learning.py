#!/usr/bin/env python
# coding: utf-8

# In[65]:


get_ipython().run_line_magic('tensorboard', '--logdir drive/My\\ Drive/Data/logs')


# In[66]:


predictions = model.predict(val_data, verbose=1) 
predictions


# In[68]:


print(predictions[0])


# In[69]:


def get_pred_label(prediction_probabilities):
  return unique_breeds[np.argmax(prediction_probabilities)]
pred_label = get_pred_label(predictions[0])
pred_label


# In[70]:


def unbatchify(data):
  images = []
  labels = []
  for image, label in data.unbatch().as_numpy_iterator():
    images.append(image)
    labels.append(unique_breeds[np.argmax(label)])
  return images, labels
val_images, val_labels = unbatchify(val_data)
val_images[0], val_labels[0]


# In[71]:


def plot_pred(prediction_probabilities, labels, images, n=1):
  pred_prob, true_label, image = prediction_probabilities[n], labels[n], images[n]
  pred_label = get_pred_label(pred_prob)
  plt.imshow(image)
  plt.xticks([])
  plt.yticks([])
  if pred_label == true_label:
    color = "green"
  else:
    color = "red"

  plt.title("{} {:2.0f}% ({})".format(pred_label,
                                      np.max(pred_prob)*100,
                                      true_label),
                                      color=color)


# In[72]:


plot_pred(prediction_probabilities=predictions,
          labels=val_labels,
          images=val_images)


# In[73]:


def plot_pred_conf(prediction_probabilities, labels, n=1):
  pred_prob, true_label = prediction_probabilities[n], labels[n]
  pred_label = get_pred_label(pred_prob)
  top_10_pred_indexes = pred_prob.argsort()[-10:][::-1]
  top_10_pred_values = pred_prob[top_10_pred_indexes]
  top_10_pred_labels = unique_breeds[top_10_pred_indexes]

  # Setup plot
  top_plot = plt.bar(np.arange(len(top_10_pred_labels)), 
                     top_10_pred_values, 
                     color="grey")
  plt.xticks(np.arange(len(top_10_pred_labels)),
             labels=top_10_pred_labels,
             rotation="vertical")
  if np.isin(true_label, top_10_pred_labels):
    top_plot[np.argmax(top_10_pred_labels == true_label)].set_color("green")
  else:
    pass


# In[74]:


plot_pred_conf(prediction_probabilities=predictions,
               labels=val_labels,
               n=9)


# In[78]:


i_multiplier = 0
num_rows = 4
num_cols = 4
num_images = num_rows*num_cols
plt.figure(figsize=(5*2*num_cols, 5*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_pred(prediction_probabilities=predictions,
            labels=val_labels,
            images=val_images,
            n=i+i_multiplier)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_pred_conf(prediction_probabilities=predictions,
                labels=val_labels,
                n=i+i_multiplier)
plt.tight_layout(h_pad=1.0)
plt.show()


# In[88]:


def save_model(model, suffix=None):
    modeldir = os.path.join("drive/My Drive/Data/models",
                            datetime.datetime.now().strftime("%Y%m%d-%H%M%f"))
    model_path = modeldir + "-" + suffix + ".h5" 
    print(f"Saving model to: {model_path}...")
    model.save(model_path)


# In[89]:





# In[90]:


save_model(model, suffix="1000-images-Adam")


# In[92]:


model.evaluate(val_data)


# In[94]:


len(x), len(y)


# In[98]:


full_data = create_data_batches(x, y)


# In[100]:


full_model = create_model()


# In[101]:


full_model_tensorboard = create_tensorboard_callback()
full_model_early_stopping = tf.keras.callbacks.EarlyStopping(monitor="accuracy",
                                                             patience=3)


# In[102]:


get_ipython().run_line_magic('tensorboard', '--logdir drive/My\\ Drive/Data/logs')


# In[ ]:




