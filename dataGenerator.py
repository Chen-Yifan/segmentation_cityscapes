from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import scipy.misc
from PIL import Image
import os
import sys
import re
import glob

# import label
labels = __import__('labels')
id2trainId = {label.id: label.trainId for label in labels.labels}  # dictionary mapping from raw IDs to train IDs
trainId2color = {label.trainId: label.color for label in labels.labels}  # dictionary mapping train IDs to colors as 3-tuples

def dataGen(train_x, train_y, val_x, val_y, batch_size):
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
    
    img_datagen.fit(train_x)
    mask_datagen.fit(train_y)

    seed = 2018
    img_gen = img_datagen.flow(train_x, seed = seed, batch_size=batch_size, shuffle=True)#shuffling
    mask_gen = mask_datagen.flow(train_y, seed = seed, batch_size=batch_size, shuffle=True)
    train_gen = zip(img_gen, mask_gen)

# val_gen
    img_datagen = ImageDataGenerator()
    mask_datagen = ImageDataGenerator()
    
    img_datagen.fit(val_x)
    mask_datagen.fit(val_y)
    
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
        for k, v in trainId2color.iteritems():
            color[label == k, :] = v
    else:
        for k, v in trainId2color.items():
            color[label == k, :] = v
    return color


def assign_trainIds(label):
    """
    Map the given label IDs to the train IDs appropriate for training
    Use the label mapping provided in labels.py from the cityscapes scripts
    """
    label = np.array(label, dtype=np.float32)
    if sys.version_info[0] < 3:
        for k, v in id2trainId.iteritems():
            label[label == k] = v
    else:
        for k, v in id2trainId.items():
            label[label == k] = v
    return label


def imgId2trainId(labelId, shape = 256):
    for a in range(shape):
        for b in range(shape):
            labelId[a,b] = assign_trainIds(labelId[a,b])
    trainId = np.where(labelId == -1, 19, labelId) # unlabeled
    trainId = np.where(labelId == 255, 19, labelId)
    return trainId


def imgTrain2color(trainId, shape = 256):
    colorimg = np.zeros((shape,shape,3))
    for a in range(shape):
        for b in range(shape):
            colorimg[a,b] = palette(trainId[a,b])
    return colorimg

def save_256(mask_path, frame_path, split, shape=256, cl=20):
    save_path = '/home/yifan/Github/segmentation_train/dataset/cityscapes_all'
    
    mask_path = os.path.join(mask_path, split)
    frame_path = os.path.join(frame_path, split)
    
    mask_files = os.listdir(mask_path)
    frame_files = os.listdir(frame_path)
    
    #sort
    frame_files.sort(key=lambda var:[int(x) if x.isdigit() else x 
                               for x in re.findall(r'[^0-9]|[0-9]+', var)])
    mask_files.sort(key=lambda var:[int(x) if x.isdigit() else x 
                               for x in re.findall(r'[^0-9]|[0-9]+', var)])
    
    # binary encode
    
    num_files = len(mask_files)
    print(len(mask_files), len(frame_files))
    
    for i in range(num_files):
        print(i)
        im = np.array(Image.open(os.path.join(mask_path, mask_files[i])))
        labelId = scipy.misc.imresize(im, (shape,shape),interp='nearest')
        trainId = imgId2trainId(labelId)
        #np.save(outfile, x)
        outfile = os.path.join(save_path, 'gtFine_256', split)
        if not os.path.isdir(outfile):
            os.makedirs(outfile)
        outfile = os.path.join(outfile, mask_files[i][:-12]+'trainIds.npy')
        print(outfile)
        np.save(outfile, trainId)
#         mask = np.eye(cl)[trainId]
        print(frame_files[i])
        im = np.array(Image.open(os.path.join(frame_path, frame_files[i])))
        frame = scipy.misc.imresize(im, (shape, shape))
        
        outfile = os.path.join(save_path, 'left_256', split)
        if not os.path.isdir(outfile):
            os.makedirs(outfile)
        outfile = os.path.join(outfile, frame_files[i][:-3]+'npy')
        print(outfile)
        np.save(outfile, frame)
#         x[i] = frame
#         y[i] = mask
                      
#     return x, y
                 
def xy_array(mask_path, frame_path, split, shape=256, cl=20):
    
    mask_path = os.path.join(mask_path, split)
    frame_path = os.path.join(frame_path, split)
    
    mask_files = os.listdir(mask_path)
    frame_files = os.listdir(frame_path)
    
    #sort
    frame_files.sort(key=lambda var:[int(x) if x.isdigit() else x 
                               for x in re.findall(r'[^0-9]|[0-9]+', var)])
    mask_files.sort(key=lambda var:[int(x) if x.isdigit() else x 
                               for x in re.findall(r'[^0-9]|[0-9]+', var)])
    
    # binary encode
    
    num_files = len(mask_files)
    print(len(mask_files), len(frame_files))
    
    x = np.zeros((num_files, shape, shape, 3)).astype(np.float32)
    y = np.zeros((num_files, shape, shape, cl)).astype(np.int8)
    
    for i in range(num_files):
#         print(i)
        img = np.load(os.path.join(frame_path, frame_files[i]))
        trainId = np.load(os.path.join(mask_path, mask_files[i]))
        mask = np.eye(cl)[trainId]
        
        x[i] = img
        y[i] = mask
    return x,y
         

def load_data(frame_path, mask_path, shape=256, cl=20):
    #training set
    train_x, train_y = xy_array(mask_path, frame_path, 'train')
    val_x, val_y = xy_array(mask_path, frame_path, 'val')
    test_x, test_y = xy_array(mask_path, frame_path, 'test')

    return train_x, train_y, val_x, val_y, test_x, test_y 

        
def save_results(test_mask_path, result_dir, test_x, test_y, predict_y):
    files = os.listdir(test_mask_path) # maintains the filename
    
    # map back to 0-19
    test_y = np.where(labelId == 19, 255, test_y)
    predict_y = np.where(labelId == 19, 255, predict_y)
    
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
    
    
    