
# coding: utf-8

# # Import images

# In[1]:


import os
import shutil
import time
import pandas as pd
import numpy as np

import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


train_filenames = os.listdir('./train/train')
train_cat = filter(lambda x:x[:3] == 'cat', train_filenames)
train_dog = filter(lambda x:x[:3] == 'dog', train_filenames)


# In[4]:


test_filenames = os.listdir('./test/test')


# In[5]:


len(train_filenames)


# In[6]:


def rmrf_mkdir(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.mkdir(dirname)


# In[70]:


# rmrf_mkdir('train2')
# os.mkdir('train2/cat')
# os.mkdir('train2/dog')

# rmrf_mkdir('test2')
# os.symlink('test/', 'test2/test')

# for filename in train_cat:
#     os.symlink('train/train/'+filename, 'train2/cat/'+filename)

# for filename in train_dog:
#     os.symlink('train/train/'+filename, 'train2/dog/'+filename)


# In[12]:


# rmrf_mkdir('train2')
# os.mkdir('train2/cat')
# os.mkdir('train2/dog')

# rmrf_mkdir('test2')

# for filename in test_filenames:
#     shutil.copy('./test/test/'+filename, './test2/')

# for filename in train_cat:
#     shutil.copy('./train/train/'+filename, './train2/cat/')

# for filename in train_dog:
#     shutil.copy('./train/train/'+filename, './train2/dog/')



# # train-validation

# In[7]:


y= list(map(lambda x:1 if x[:3] == 'dog' else 0, train_filenames))


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


X_train, X_val, y_train, y_val = train_test_split(train_filenames, y, test_size=0.2, random_state=20277)


# In[10]:


len([filename for filename in X_train if filename[:3] == 'cat'])


# In[11]:


len([filename for filename in X_train if filename[:3] == 'dog'])


# In[185]:


# rmrf_mkdir('train_img')
# os.mkdir('train_img/cat')
# os.mkdir('train_img/dog')

# for filename in X_train:
#     if filename[:3] == 'cat':
#         shutil.copy('./train/train/'+filename, './train_img/cat/')
#     else:
#         shutil.copy('./train/train/'+filename, './train_img/dog/')


# In[186]:


# rmrf_mkdir('val_img')
# os.mkdir('val_img/cat')
# os.mkdir('val_img/dog')


# for filename in X_val:
#     if filename[:3] == 'cat':
#         shutil.copy('./train/train/'+filename, './val_img/cat/')
#     else:
#         shutil.copy('./train/train/'+filename, './val_img/dog/')


# # Fine-tune InceptionV3

# In[12]:


from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import GlobalAveragePooling2D,Dense,Dropout
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras import optimizers
from keras.callbacks import EarlyStopping


# In[13]:


image_size=(299,299)


# In[14]:


# gen = ImageDataGenerator(zoom_range=[0.8,1.2],
#                          rotation_range=10,
#                          width_shift_range=0.2,
#                          height_shift_range=0.2,
#                          preprocessing_function=preprocess_input)


# In[15]:


gen = ImageDataGenerator(preprocessing_function=preprocess_input,       
                            rotation_range=30,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True)


# In[16]:


bs=64


# In[17]:


train_generator = gen.flow_from_directory("./train_img", image_size, shuffle=False, batch_size=bs,class_mode= "binary")


# In[18]:


val_generator = gen.flow_from_directory("./val_img", image_size, shuffle=False, batch_size=bs,class_mode= "binary")


# In[19]:


# create the base pre-trained model
InceptionV3_base_model = InceptionV3(weights='imagenet', include_top=False)


# In[20]:


InceptionV3_x = InceptionV3_base_model.output


# In[21]:


InceptionV3_x = GlobalAveragePooling2D()(InceptionV3_x)


# In[22]:


InceptionV3_x = Dense(1024, activation='relu')(InceptionV3_x)


# In[23]:


InceptionV3_x = Dropout(0.4)(InceptionV3_x)


# In[24]:


InceptionV3_predictions = Dense(1, activation='sigmoid')(InceptionV3_x)


# In[25]:


InceptionV3_model = Model(inputs = InceptionV3_base_model.input, outputs = InceptionV3_predictions)


# In[26]:


for layer in InceptionV3_base_model.layers:
    layer.trainable = False


# In[27]:


InceptionV3_model.compile(optimizer=optimizers.SGD(lr=0.005, momentum=0.9), loss='binary_crossentropy',metrics=['accuracy'])


# In[28]:


InceptionV3_history_ = InceptionV3_model.fit_generator(train_generator,steps_per_epoch= len(X_train)/bs, epochs=5,verbose=1, validation_data=val_generator, validation_steps=len(X_val)/bs)


# In[29]:


InceptionV3_model.save('InceptionV3_base_model.h5')


# In[30]:


# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:

for layer in InceptionV3_model.layers[:249]:
    layer.trainable = False
for layer in InceptionV3_model.layers[249:]:
    layer.trainable = True   


# In[31]:


model_early_stop=EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')


# In[32]:


InceptionV3_model.compile(optimizer=optimizers.SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy',metrics=['accuracy'])


# In[ ]:


InceptionV3_history = InceptionV3_model.fit_generator(train_generator, callbacks=[model_early_stop],steps_per_epoch= len(X_train)/bs, epochs=10,verbose=1, validation_data=val_generator, validation_steps=len(X_val)/bs)


# In[ ]:


InceptionV3_model.save('InceptionV3_model.h5')


# In[61]:


def plot_training(history): 
#     acc = history.history['acc'] 
#     val_acc = history.history['val_acc'] 
    loss = history.history['loss'] 
    val_loss = history.history['val_loss'] 
    epochs = range(len(loss)) 
#     plt.plot(epochs, acc, 'r.') 
#     plt.plot(epochs, val_acc, 'r') 
#     plt.title('Training and validation accuracy') 
#     plt.figure() 
    plt.plot(epochs, loss, 'b-') 
    plt.plot(epochs, val_loss, 'r-') 
    plt.title('Training and validation loss') 
    plt.show()


# In[62]:


plot_training(InceptionV3_history)


    train = model.predict_generator(train_generator, train_generator.nb_sample)
    test = model.predict_generator(test_generator, test_generator.nb_sample)
    with h5py.File("gap_%s.h5"%MODEL.func_name) as h:
        h.create_dataset("train", data=train)
        h.create_dataset("test", data=test)
        h.create_dataset("label", data=train_generator.classes)

write_gap(ResNet50, (224, 224))
write_gap(InceptionV3, (299, 299), inception_v3.preprocess_input)
write_gap(Xception, (299, 299), xception.preprocess_input)

