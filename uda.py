
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


# # Exploration

# In[60]:


random.seed(13)


# In[61]:


indices = random.sample(range(len(train_filenames)), 9)


# In[62]:


indices 


# In[63]:


plt.figure(figsize=(20,20))
for i in range(9):
    plt.subplot(3,3, i+1)
    plt.imshow(mpimg.imread('./train/train/%s'%train_filenames[indices[i]]))
    plt.title("%s"%train_filenames[indices[i]])


# # Check 'outliers'

# In[6]:


dogs = [
 'n02085620','n02085782','n02085936','n02086079'
,'n02086240','n02086646','n02086910','n02087046'
,'n02087394','n02088094','n02088238','n02088364'
,'n02088466','n02088632','n02089078','n02089867'
,'n02089973','n02090379','n02090622','n02090721'
,'n02091032','n02091134','n02091244','n02091467'
,'n02091635','n02091831','n02092002','n02092339'
,'n02093256','n02093428','n02093647','n02093754'
,'n02093859','n02093991','n02094114','n02094258'
,'n02094433','n02095314','n02095570','n02095889'
,'n02096051','n02096177','n02096294','n02096437'
,'n02096585','n02097047','n02097130','n02097209'
,'n02097298','n02097474','n02097658','n02098105'
,'n02098286','n02098413','n02099267','n02099429'
,'n02099601','n02099712','n02099849','n02100236'
,'n02100583','n02100735','n02100877','n02101006'
,'n02101388','n02101556','n02102040','n02102177'
,'n02102318','n02102480','n02102973','n02104029'
,'n02104365','n02105056','n02105162','n02105251'
,'n02105412','n02105505','n02105641','n02105855'
,'n02106030','n02106166','n02106382','n02106550'
,'n02106662','n02107142','n02107312','n02107574'
,'n02107683','n02107908','n02108000','n02108089'
,'n02108422','n02108551','n02108915','n02109047'
,'n02109525','n02109961','n02110063','n02110185'
,'n02110341','n02110627','n02110806','n02110958'
,'n02111129','n02111277','n02111500','n02111889'
,'n02112018','n02112137','n02112350','n02112706'
,'n02113023','n02113186','n02113624','n02113712'
,'n02113799','n02113978']

cats=[
'n02123045','n02123159','n02123394','n02123597'
,'n02124075','n02125311','n02127052']


# In[11]:


from tqdm import tqdm
import cv2
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions


# In[21]:


# n =100

# X = np.zeros((n, 224, 224, 3), dtype=np.float32)
# for i in tqdm(range(n)):
#     X[i] = cv2.resize(cv2.imread('./train/train/%s'%train_filenames[i]), (224, 224))


# In[102]:


ResNet50_tp5 =[]


# In[103]:


import time


# In[104]:


start_time =time.time()


# In[107]:


n =10000


# In[ ]:


model = ResNet50(weights='imagenet')


# In[105]:


for i in range(n):
    img = image.load_img('./train/train/%s'%train_filenames[i], target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    rst = list(pd.DataFrame(decode_predictions(preds, top=30)[0])[0])
    if [c for c in rst if c in dogs]  !=[] :
        if [c for c in rst if c in cats]  !=[] :
            ResNet50_tp5.append(2)
        else:
            ResNet50_tp5.append(1)
    elif [c for c in rst if c in cats]  !=[] :
            ResNet50_tp5.append(0)
    else :
            ResNet50_tp5.append(-1)
cost_time = time.time() - start_time


# In[106]:


print("--- %s seconds ---" % (cost_time))


# In[108]:


ResNet50_tp5_df = pd.Series(ResNet50_tp5,index = range(n))


# In[109]:


ResNet50_tp5_df[ResNet50_tp5_df==-1].index


# In[110]:


i_ = 1185


# In[111]:


plt.imshow(mpimg.imread('./train/train/%s'%train_filenames[i_]))
plt.title("%s"%train_filenames[i_])


# In[ ]:


from keras.applications.resnet50 import Xception
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np


# In[6]:


model = ResNet50(weights='imagenet')


# In[15]:


start_time =time.time()


# In[ ]:


for i in range(len(train_filenames)):
    img = image.load_img('./train/train/%s'%train_filenames[i], target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    ResNet50_tp3.append(decode_predictions(preds, top=3)[0])
cost_time = time.time() - start_time


# In[ ]:


del model


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


# In[107]:


InceptionV3_history = InceptionV3_model.fit_generator(train_generator, callbacks=[model_early_stop],steps_per_epoch= len(X_train)/bs, epochs=10,verbose=1, validation_data=val_generator, validation_steps=len(X_val)/bs)


# In[38]:


InceptionV3_history = InceptionV3_model.fit_generator(train_generator, steps_per_epoch= len(X_train)/bs, epochs=30,verbose=1, validation_data=val_generator, validation_steps=len(X_val)/bs)


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


# In[ ]:


# Visualize training history
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
history = model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


InceptionV3_model.save('InceptionV3_model.h5')


# In[88]:


for i, layer in enumerate(InceptionV3_base_model.layers):
    print(i, layer.name)


# In[48]:


from keras.models import load_model


# In[ ]:


# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers from inception V3. We will freeze the bottom N layers
# and train the remaining top layers.

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:249]:
    layer.trainable = False
for layer in model.layers[249:]:
    layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
from keras.optimizers import SGD
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy')

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
model.fit_generator(...)


# # submission

# In[134]:


test_generator = gen.flow_from_directory("./test", image_size, shuffle=False, batch_size=32,class_mode= "binary")


# In[135]:


y_pred = InceptionV3_model.predict_generator(test_generator, verbose=1)


# In[136]:


y_pred 


# In[137]:


y_pred = y_pred.clip(min=0.005, max=0.995)


# In[146]:


y_pred 


# In[162]:


import pandas as pd
# from keras.preprocessing.image import *

df = pd.read_csv("./sample_submission.csv")


# In[164]:


for i, fname in enumerate(test_generator.filenames):
    index = int(fname[fname.rfind('/')+1:fname.rfind('.')])
    df.loc[index-1,'label'] = y_pred[i][0]


# In[166]:


df.to_csv('pred.csv', index=None)


# ## ypw

# In[4]:


from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
import h5py


# In[7]:


image_size=(299,299)
width = image_size[0]
height = image_size[1]
input_tensor = Input((height, width, 3))
x = input_tensor


# In[5]:


if lambda_func:
    x = Lambda(lambda_func)(x)


# In[8]:


base_model = InceptionV3(input_tensor=x, weights='imagenet', include_top=False)


# In[10]:


model = InceptionV3(input_tensor = base_model.input, pooling= GlobalAveragePooling2D()(base_model.output))


# In[16]:


gen = ImageDataGenerator()
train_generator = gen.flow_from_directory("train2", image_size, shuffle=False, 
                                          batch_size=16)
test_generator = gen.flow_from_directory("test2", image_size, shuffle=False, 
                                         batch_size=16, class_mode=None)


# In[17]:


train = model.predict_generator(train_generator, train_generator.nb_sample)


# In[ ]:


test = model.predict_generator(test_generator, test_generator.nb_sample)


# In[ ]:


with h5py.File("gap_%s.h5"%MODEL.func_name) as h:
    h.create_dataset("train", data=train)
    h.create_dataset("test", data=test)
    h.create_dataset("label", data=train_generator.classes)


# In[4]:




def write_gap(MODEL, image_size, lambda_func=None):
    width = image_size[0]
    height = image_size[1]
    input_tensor = Input((height, width, 3))
    x = input_tensor
    if lambda_func:
        x = Lambda(lambda_func)(x)

    base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

    gen = ImageDataGenerator()
    train_generator = gen.flow_from_directory("train2", image_size, shuffle=False, 
                                              batch_size=16)
    test_generator = gen.flow_from_directory("test2", image_size, shuffle=False, 
                                             batch_size=16, class_mode=None)

    train = model.predict_generator(train_generator, train_generator.nb_sample)
    test = model.predict_generator(test_generator, test_generator.nb_sample)
    with h5py.File("gap_%s.h5"%MODEL.func_name) as h:
        h.create_dataset("train", data=train)
        h.create_dataset("test", data=test)
        h.create_dataset("label", data=train_generator.classes)

write_gap(ResNet50, (224, 224))
write_gap(InceptionV3, (299, 299), inception_v3.preprocess_input)
write_gap(Xception, (299, 299), xception.preprocess_input)

