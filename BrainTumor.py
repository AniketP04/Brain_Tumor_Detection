#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import keras
from matplotlib import pyplot as plt
import numpy as np
import gzip
get_ipython().run_line_magic('matplotlib', 'inline')
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, ModelCheckpoint
from sklearn.metrics import accuracy_score


# In[ ]:


from google.colab import drive
drive.mount('/content/gdrive')


# In[ ]:


#model parameters
batch_size = 64
epochs = 2000
inChannel = 1
iml, imh = 128, 128
input_img = Input(shape = (iml, imh, inChannel))
num_classes = 3
batch = 128


# In[ ]:


import os
from PIL import Image
for i in range(3):
  path = 'gdrive/My Drive/MRIBrain/train/' + str(i)
  files = os.listdir(path)
  i = 1

  for file in files:
      im = Image.open(os.path.join(path, file))
  #     img = im.crop((0, 0, 450, 375))
      img = im.crop((55, 55, 400, 375))
      img.save(os.path.join(path, file))


# In[ ]:


def AHE(img):
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
    return img_adapteq
train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',preprocessing_function=AHE)

train_generator=train_datagen.flow_from_directory('gdrive/My Drive/MRIBrain/train',
                                                 target_size=(iml,imh),
                                                 color_mode='grayscale',
                                                 batch_size=batch,
                                                 class_mode='input',
                                                 shuffle=True,)


# In[ ]:


val_datagen = ImageDataGenerator(rescale=1. /255)
val_generator=train_datagen.flow_from_directory('gdrive/My Drive/MRIBrain/val',
                                                 target_size=(iml,imh),
                                                 color_mode='grayscale',
                                                 batch_size=batch,
                                                 class_mode='input',
                                                 shuffle=True)


# In[ ]:


def encoder(input_img):
    #encoder

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 256 (small and thick)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    return conv4



# In[ ]:


def decoder(conv4):    
    #decoder
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4) #7 x 7 x 128
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5) #7 x 7 x 64
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    up1 = UpSampling2D((2,2))(conv6) #14 x 14 x 64
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 32
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    up2 = UpSampling2D((2,2))(conv7) # 28 x 28 x 32
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    return decoded


# In[ ]:


autoencoder = Model(input_img, decoder(encoder(input_img)))
autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())


# In[ ]:


autoencoder.summary()


# In[ ]:


lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=8)
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, mode='auto')
checkpointer = ModelCheckpoint('gdrive/My Drive/101/{epoch:02d}-{val_loss:.4f}.hdf5', mode='min', monitor='val_loss', verbose=1, save_best_only=True)


# In[ ]:


batch_size = batch
learning_rate = 0.001


# In[ ]:


autoencoder.load_weights('gdrive/My Drive/101/06-0.0806.hdf5')


# In[ ]:


# checkpoint = ModelCheckpoint('gdrive/My Drive/101/{epoch:02d}-{val_acc:.2f}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# callbacks_list = [checkpoint]
step_size_train=train_generator.n//train_generator.batch_size
step_size_val=val_generator.n//val_generator.batch_size
model_history = []
model_history.append(autoencoder.fit_generator(generator=train_generator,
                    validation_data=val_generator,
                   steps_per_epoch=step_size_train,
                                               
                    validation_steps = step_size_val,
                                               
                 epochs=epochs,shuffle = True,
          callbacks=[lr_reducer, checkpointer, early_stopper],initial_epoch =0))


# In[ ]:


import matplotlib.pyplot as plt
plt.title('Accuracies vs Epochs')
plt.plot(model_history[0].history['acc'], label='Training_Accuracy')
plt.plot(model_history[0].history['val_acc'], label='validation Accuacy')
plt.plot(model_history[0].history['loss'], label='Training_loss')
plt.plot(model_history[0].history['val_loss'], label='Val_loss')
plt.legend()
plt.show()


# Classifier

# 

# In[ ]:


def fc(enco):
    flat = Flatten()(enco)
    den = Dense(128, activation='relu')(flat)
    out = Dense(num_classes, activation='softmax')(den)
    return out


# In[ ]:


encode = encoder(input_img)
full_model = Model(input_img,fc(encode))


# In[ ]:


for l1,l2 in zip(full_model.layers[:19],autoencoder.layers[0:19]):
    l1.set_weights(l2.get_weights())


# In[ ]:


autoencoder.get_weights()[0][1]


# In[ ]:


full_model.get_weights()[0][1]


# In[ ]:


for layer in full_model.layers[0:19]:
    layer.trainable = False


# In[ ]:


full_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),metrics=['accuracy'])


# In[ ]:


full_model.summary()


# In[ ]:


def AHE(img):
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
    return img_adapteq
train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',preprocessing_function=AHE)

train_generator=train_datagen.flow_from_directory('gdrive/My Drive/MRIBrain/train',
                                                 target_size=(iml,imh),
                                                 color_mode='grayscale',
                                                 batch_size=batch,
                                                 class_mode='categorical',
                                                 shuffle=True)


# In[ ]:


val_datagen = ImageDataGenerator(rescale=1. /255,preprocessing_function=AHE)
val_generator=train_datagen.flow_from_directory('gdrive/My Drive/MRIBrain/val',
                                                 target_size=(iml,imh),
                                                 color_mode='grayscale',
                                                 batch_size=batch,
                                                 class_mode='categorical',
                                                 shuffle=False)


# In[ ]:


lr_reducer_FULL = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=8)
early_stopper_FULL = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, mode='auto')
checkpointer_FULL = ModelCheckpoint('gdrive/My Drive/101/FULL{epoch:02d}-{val_loss:.4f}.hdf5', mode='min', monitor='val_loss', verbose=1, save_best_only=True)


# In[ ]:


step_size_train=train_generator.n//train_generator.batch_size
step_size_val=val_generator.n//val_generator.batch_size
full_model_history = []
full_model_history.append(full_model.fit_generator(generator=train_generator,
                                                   validation_data=val_generator,
                                                   steps_per_epoch=step_size_train,
                                                   validation_steps = step_size_val,
                                                   epochs=epochs,shuffle = True,
                                                   callbacks=[lr_reducer_FULL, checkpointer_FULL, early_stopper_FULL],initial_epoch =0))


# In[ ]:


import matplotlib.pyplot as plt
plt.title('Accuracies vs Epochs')
plt.plot(full_model_history[0].history['acc'], label='Training_Accuracy')
plt.plot(full_model_history[0].history['val_acc'], label='validation Accuacy')
plt.plot(full_model_history[0].history['loss'], label='Training_loss')
plt.plot(full_model_history[0].history['val_loss'], label='Val_loss')
plt.legend()
plt.show()


# In[ ]:


plt.plot(full_model_history[0].history['acc'], label='Training_Accuracy')
plt.plot(full_model_history[0].history['val_acc'], label='validation Accuacy')
plt.legend()
plt.show()


# In[ ]:




