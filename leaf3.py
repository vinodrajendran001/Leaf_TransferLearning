import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.data_utils import get_file
from keras import optimizers
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import backend as K

K.set_image_dim_ordering('th')

import pandas as pd
import os, cv2, random, re
import numpy as np
np.random.seed(100)

# data processing, CSV file I/O (e.g. pd.read_csv)
input1 = pd.read_csv('../leafclassification/train.csv')
parent_data = input1.copy() # for submission
input2 = pd.read_csv('../leafclassification/test.csv')
index = input2['id'] # for submission
targets = input1['species']
train_id = input1['id']
test_id = input2['id']
print len(targets)
print train_id[0]
print test_id[0]
# Formatting the images for CNN input using openCV

CHANNELS = 3
ROWS = 64
COLS = 64

TRAIN_DIR = 'images/'
train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)]
print len(train_images)

def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
    return cv2.resize(img, (ROWS, COLS), interpolation=cv2.INTER_CUBIC)

def prep_data(images,train):
    count = len(images)+1
    targets_id=[]
    data = np.ndarray((count, CHANNELS, ROWS, COLS), dtype=np.uint8)
    for i, image_file in enumerate(images):
        image = read_image(image_file)
        if train == 1:
            if int(re.findall('\d+', image_file)[0]) in train_id.values:
                data[int(re.findall('\d+', image_file)[0])] = image.T
                targets_id.append(int(re.findall('\d+', image_file)[0]))
                #print(int(re.findall('\d+', image_file)[0]))
        if train == 0:
            if int(re.findall('\d+', image_file)[0]) in test_id.values:
                data[int(re.findall('\d+', image_file)[0])] = image.T
        if i%250 == 0:
            print('Processed {} of {}'.format(i, count))
    if train == 1:
        return data,targets_id
    if train == 0:
        return data


train, targets_id = prep_data(train_images, 1)
test = prep_data(train_images, 0)

print("Train shape: {}".format(train.shape))
print("Test shape: {}".format(test.shape))

train_data = np.ndarray((len(train_id), 3, ROWS, COLS), dtype=np.uint8)
count = 0

for i in train_id:
    train_data[count] = train[i]
    count = count + 1
test_data = np.ndarray((len(test_id), 3, ROWS, COLS), dtype=np.uint8)
count = 0
for i in test_id:
    test_data[count] = test[i]
    count = count + 1

print("Train shape: {}".format(train_data.shape))
print("Test shape: {}".format(test_data.shape))

# converting target labels into categorical values

from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical

y_train = targets
y_train = LabelEncoder().fit(y_train).transform(y_train)
print(y_train.shape)
y_train_cat = to_categorical(y_train)
print(y_train_cat)
print train_data.shape


# path to the model weights file.
TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
top_model_weights_path = 'bottleneck_fc_model_bk.h5' #output of leaf2.py
# dimensions of our images.

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = train_data.shape[0]
nb_validation_samples = test_data.shape[0]
nb_epoch = 100

# build the VGG16 network
model = Sequential()
model.add(ZeroPadding2D((1, 1), input_shape=(3, ROWS, COLS)))

model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# load the weights of the VGG16 networks
# (trained on ImageNet, won the ILSVRC competition in 2014)
# note: when there is a complete match between your model definition
# and your weight savefile, you can simply call model.load_weights(filename)

# use if not available in local
weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels.h5',
                        TF_WEIGHTS_PATH, cache_subdir='models')

# weights_path = "weights/vgg16_weights.h5" # use if its not available in local
f = h5py.File(weights_path)
for k in range(f.attrs['nb_layers']):
    if k >= len(model.layers):
        # we don't look at the last (fully-connected) layers in the savefile
        break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    model.layers[k].set_weights(weights)
f.close()
print('Model loaded.')

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(99, activation='sigmoid'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
model.add(top_model)

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:25]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# prepare data augmentation configuration

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_datagen.fit(train_data)

checkpointer = ModelCheckpoint(filepath="weights_leaf3.h5", monitor='loss',
                               verbose=1, save_best_only=True)

# Fit the model on the batches generated by train_datagen.

model.fit_generator(train_datagen.flow(train_data, y_train_cat,
                    batch_size=32),
                    samples_per_epoch=train_data.shape[0],
                    nb_epoch=nb_epoch, verbose=1,
                    callbacks=[TensorBoard(log_dir='leaf3'), checkpointer]
                    )

yPred = model.predict_proba(test_data)
yPred = pd.DataFrame(yPred,index=index, columns=sorted(parent_data.species.unique()))
fp = open('submission_nn_kernel_leaf3.csv','w')
fp.write(yPred.to_csv())
