from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import scipy.misc
from PIL import Image
import os
import sys
from sklearn.preprocessing import OneHotEncoder

# import label
labels = __import__('labels')
id2trainId = {label.id: label.trainId for label in labels.labels}  # dictionary mapping from raw IDs to train IDs
trainId2color = {label.trainId: label.color for label in labels.labels}  # dictionary mapping train IDs to colors as 3-tuples

def dataGen(train_x, train_y, val_x, val_y, batch_size):
    
#     datagen = ImageDataGenerator(
#     featurewise_center=True,
# #     featurewise_std_normalization=True,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True)

# train gen
    data_gen_args = dict(
#                         horizontal_flip = True,
#                          vertical_flip = True,
                         width_shift_range = 0.1,
                         height_shift_range = 0.1,
                         zoom_range = 0.1,
                         rotation_range = 10,
                         featurewise_center=True,
                        )
    img_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    seed = 2018
    img_gen = img_datagen.flow(train_x, seed = seed, batch_size=batch_size, shuffle=True)#shuffling
    mask_gen = mask_datagen.flow(train_y, seed = seed, batch_size=batch_size, shuffle=True)
    train_gen = zip(img_gen, mask_gen)

# val_gen
    img_datagen = ImageDataGenerator()
    mask_datagen = ImageDataGenerator()
            
    img_gen = img_datagen.flow(val_x, batch_size=batch_size, shuffle=True)
    mask_gen = mask_datagen.flow(val_y, batch_size=batch_size, shuffle=True)
    val_gen = zip(img_gen, mask_gen)    
        
    return train_gen, val_gen

def palette(label):
    '''
    Map trainIds to colors as specified in labels.py
    '''
    if label.ndim == 3:
        label= label[0]
    color = np.empty((256, 256, 3))
    if sys.version_info[0] < 3:
        for k, v in self.trainId2color.iteritems():
            color[label == k, :] = v
    else:
        for k, v in self.trainId2color.items():
            color[label == k, :] = v
    return color


def assign_trainIds(label):
    """
    Map the given label IDs to the train IDs appropriate for training
    Use the label mapping provided in labels.py from the cityscapes scripts
    """
    label = np.array(label, dtype=np.float32)
    if sys.version_info[0] < 3:
        for k, v in self.id2trainId.iteritems():
            label[label == k] = v
    else:
        for k, v in self.id2trainId.items():
            label[label == k] = v
    return label


def imgId2trainId(labelId, shape = 256):
    for a in range(shape):
        for b in range(shape):
            labelId[a,b] = assign_trainIds(labelId[a,b])
    trainId = np.where(labelId = 255, 19, labelId)
    trainId = np.where(labelId = -1, 19, labelId) # unlabeled
    return trainId


def imgTrain2color(trainId, shape = 256):
    colorimg = np.zeros((shape,shape,3))
    for a in range(shape):
        for b in range(shape):
            colorimg[a,b] = palette(trainId[a,b])
    return colorimg


def load_trainval(train_mask_path, train_frame_path, shape=256, cl=20):
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
#     onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    
    files = os.listdir(train_mask_path)
    train_split = int(0.8*len(files))
    train_files = files[:train_split]
    val_files = files[train_split:]
    
    #training set
    train_x = np.zeros((len(train_files), shape, shape, 3)).astype(np.float32)
    train_y = np.zeros((len(train_files), shape, shape, cl)).astype(np.int8)
    
    for i in range(len(train_files)): 
        im = np.array(Image.open(os.path.join(train_mask_path, train_files[i])))
        labelId = scipy.misc.imresize(im, (shape, shape),interp='nearest')
        print(labelId[0,0])
        trainId = imgId2trainId(labelId)
        mask = onehot_encoder.fit_transform(trainId)
        
        im = np.array(Image.open(os.path.join(train_frame_path, train_files[i])))
        frame = scipy.misc.imresize(im, (shape, shape))
        
        train_x[i] = frame
        train_y[i] = mask
        
    #validation set
    val_x = np.zeros((len(val_files), shape, shape, 3)).astype(np.float32)
    val_y = np.zeros((len(val_files), shape, shape, cl)).astype(np.int8)

    for i in range(len(val_files)): 
        im = np.array(Image.open(os.path.join(train_mask_path, val_files[i])))
        labelId = scipy.misc.imresize(im, (shape, shape),interp='nearest')
        trainId = imgId2trainId(labelId)
        mask = onehot_encoder.fit_transform(trainId)
        
        im = np.array(Image.open(os.path.join(train_frame_path, val_files[i])))
        frame = scipy.misc.imresize(im, (shape, shape))
        
        val_x[i] = frame
        val_y[i] = mask
        
    return train_x, train_y, val_x, val_y

def load_test(test_mask_path, test_frame_path, shape=256, cl=20):
    
    files = os.listdir(test_mask_path)
    test_x = np.zeros((len(files), shape, shape, 3)).astype(np.float32)
    test_y = np.zeros((len(files), shape, shape, cl)).astype(np.int8)
    
    for i in range(len(files)): 
        im = np.array(Image.open(os.path.join(test_mask_path,files[i])))
        labelId = scipy.misc.imresize(im, (shape, shape))
        trainId = imgId2trainId(labelId)
        mask = onehot_encoder.fit_transform(trainId)
        
        im = np.array(Image.open(os.path.join(test_frame_path,files[i])))
        frame = scipy.misc.imresize(im, (shape, shape))
        
        val_x[i] = frame
        val_y[i] = mask
    return test_x, test_y

        
def save_results(test_mask_path, result_dir, test_x, test_y, predict_y):
    files = os.listdir(test_mask_path) # maintains the filename
    
    # map back to 0-19
    test_y = np.where(labelId = 19, 255, test_y)
    predict_y = np.where(labelId = 19, 255, predict_y)
    
    for i in range(len(files)):
        # 256,256,1 -- id --> change to color
        gt = imgTrain2color(np.argmax(test_y))
        pre = imgTrain2color(np.argmax(predict_y))

        im = Image.fromarray(test_x)
        im.save(files[i][:-4] + '_A.jpg')
        
        im = Image.fromarray(gt)
        im.save(files[i][:-4] + '_B.jpg')
        
        im = Image.fromarray(pre)
        im.save(files[i][:-4] + '_pre.jpg')
    
    
    