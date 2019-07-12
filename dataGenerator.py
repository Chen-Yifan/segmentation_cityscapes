from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import scipy.misc
from PIL import Image
import os
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


def load_trainval(train_mask_path, train_frame_path, shape=256, cl=20):
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
#     onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    
    files = os.listdir(train_mask_path)
    train_split = int(0.7*len(files))
    train_files = files[:train_split]
    val_files = files[train_split:]
    
    #training set
    train_x = np.zeros((len(train_files), shape, shape, 3)).astype(np.float32)
    train_y = np.zeros((len(train_files), shape, shape, cl)).astype(np.int8)
    
    for i in range(len(train_files)): 
        im = np.array(Image.open(os.path.join(train_mask_path, train_files[i])))
        labelId = scipy.misc.imresize(im, (shape, shape),interp='nearest')
        print(labelId[0,0])
        for a in range(shape):
            for b in range(shape):
                labelId[a,b] = id2trainId[labelId[a,b]]
        trainId = np.where(labelId = 255, 19, labelId)
        trainId = np.where(labelId = -1, 19, labelId)
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
        for a in range(shape):
            for b in range(shape):
                labelId[a,b] = id2trainId[labelId[a,b]]
        trainId = np.where(labelId = 255, 19, labelId)
        trainId = np.where(labelId = -1, 19, labelId) # unlabeled
        
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
    
    for file in files: 
        im = np.array(Image.open(os.path.join(test_mask_path,file)))
        labelId = scipy.misc.imresize(im, (shape, shape))
        for a in range(shape):
            for b in range(shape):
                labelId[a,b] = id2trainId[labelId[a,b]]
        trainId = np.where(labelId = 255, 19, labelId)
        trainId = np.where(labelId = -1, 19, labelId) # unlabeled
        mask = onehot_encoder.fit_transform(trainId)
        
        im = np.array(Image.open(os.path.join(test_frame_path,file)))
        frame = scipy.misc.imresize(im, (shape, shape))
        
        val_x[i] = frame
        val_y[i] = mask
    return test_x, test_y