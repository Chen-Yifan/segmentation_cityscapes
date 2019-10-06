from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import scipy.misc
from PIL import Image
import imageio
import os
import sys
import re
import glob
from utils import *

# import label
labels = __import__('labels')
id2trainId = {label.id: label.trainId for label in labels.labels}  # dictionary mapping from raw IDs to train IDs
trainId2color = {label.trainId: label.color for label in labels.labels}  # dictionary mapping train IDs to colors as 3-tuples
id2color = {label.id: label.color for label in labels.labels} 


def palette(label,shape=256):
    '''
    Map labelIds to colors as specified in labels.py
    '''
#     print(label.shape, label.ndim)
    if label.ndim == 3:
        label= label[0]
#         print('yes')
    color = np.empty((shape, shape, 3),dtype=int)
    if sys.version_info[0] < 3:
        for k, v in id2color.iteritems():
            color[label == k, :] = v
    else:
        for k, v in id2color.items():
            color[label == k, :] = v
    return color


def load_data(frame_path, mask_path, shape=256, cl=34):
    #training set
    val_x, val_y = xy_formarray(mask_path, frame_path, 'val', shape, cl)
    train_x, train_y = xy_formarray(mask_path, frame_path, 'train', shape, cl)
#     test_x, test_y = xy_formarray(mask_path, frame_path, 'test')

    return train_x, train_y, val_x, val_y

def xy_formarray(mask_path, frame_path, split, shape=256, cl=34):
    
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
    y = np.zeros((num_files, shape, shape, cl)).astype(np.uint8)
    
    for i in range(num_files):
#         print(i)
        img = np.load(os.path.join(frame_path, frame_files[i]))
        mask = np.load(os.path.join(mask_path, mask_files[i]))# (256,256) 
        mask = np.eye(cl)[mask]
        
        x[i] = img
        y[i] = mask
    return x,y
         

def trainGen(train_x, train_y, batch_size, cl=34):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    #has rescaled when loading the data
    x_gen_args = dict(
                    rescale = 1./255,
                    rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
    
    y_gen_args = dict(
                    rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
    
    img_datagen = ImageDataGenerator(**x_gen_args)
    mask_datagen = ImageDataGenerator(**y_gen_args)
    
    img_datagen.fit(train_x)
    mask_datagen.fit(train_y)

    seed = 2018
    img_gen = img_datagen.flow(train_x, seed = seed, batch_size=batch_size, shuffle=True)#shuffling
    mask_gen = mask_datagen.flow(train_y, seed = seed, batch_size=batch_size, shuffle=True)
    
    train_gen = zip(img_gen, mask_gen)

    return train_gen


def testGen(val_x, val_y, batch_size):
# val_gen
    img_datagen = ImageDataGenerator()
    mask_datagen = ImageDataGenerator()
    
    img_datagen.fit(val_x)
    mask_datagen.fit(val_y)
    
    seed = 1
    img_gen = img_datagen.flow(val_x, seed = seed, batch_size=batch_size, shuffle=True)
    mask_gen = mask_datagen.flow(val_y, seed = seed, batch_size=batch_size, shuffle=True)
    val_gen = zip(img_gen, mask_gen)    
        
    return val_gen


def save_results(mask_path, result_dir, test_x, test_y, predict_y, split='test'):
    test_mask_path = os.path.join(mask_path,split)
    files = os.listdir(test_mask_path) # maintains the filename
    
    # map back to 0-33
    test_y = np.argmax(test_y, axis=-1)
    predict_y = np.argmax(predict_y, axis=-1).astype(np.uint8)
    
    for i in range(len(files)):
        # 256,256,1 -- id --> change to color
        label_gt = test_y[i]
        color_gt = palette(label_gt)
        label_pred = predict_y[i]
        color_pred = palette(label_pred)
        
        imageio.imwrite(os.path.join(result_dir, files[i][:-20] + '_A.jpg'), test_x[i].astype('uint8'))
        imageio.imwrite(os.path.join(result_dir, files[i][:-20] + '_B_color_gt.png'), color_gt.astype('uint8'))
        imageio.imwrite(os.path.join(result_dir,files[i][:-20] + '_B_color_pred.png'), color_pred.astype('uint8'))
        
        label_gt = scipy.misc.imresize(label_gt, (1024,2048),interp='nearest')
        label_pred = scipy.misc.imresize(label_pred, (1024,2048),interp='nearest')
        imageio.imwrite(os.path.join(result_dir, files[i][:-20] + '_B_label_gt.png'), label_gt)
        imageio.imwrite(os.path.join(result_dir,files[i][:-20] + '_B_label_pred.png'), label_pred)
        

    
    
