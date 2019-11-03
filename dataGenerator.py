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
import cv2

# import label
labels = __import__('labels')
id2trainId = {label.id: label.trainId for label in labels.labels}  # dictionary mapping from raw IDs to train IDs
trainId2color = {label.trainId: label.color for label in labels.labels}  # dictionary mapping train IDs to colors as 3-tuples
id2color = {label.id: label.color for label in labels.labels} 


def palette(label,shape):
    '''
    Map labelIds to colors as specified in labels.py
    '''
    if label.ndim == 3:
        label= label[0]
    color = np.empty((shape[0], shape[1], 3),dtype=int)
    if sys.version_info[0] < 3:
        for k, v in id2color.iteritems():
            color[label == k, :] = v
    else:
        for k, v in id2color.items():
            color[label == k, :] = v
    return color


def load_test(mask_path, frame_path, split, recursive=True, shape=(512,1024), name_list=False):
    '''
        from png to npy array with reshape = shape x shape
        
        no augmentation
        
        only for test and validation: resize to 512x1024
    '''
    mask_path = os.path.join(mask_path, split)
    frame_path = os.path.join(frame_path, split)
    frame_names = []
    if(recursive):
        x = []
        y = []
        cities = os.listdir(mask_path)
        # recursively load the png files
        for city in cities:
            print(city)
            mask_file = os.path.join(mask_path, city)
            frame_file = os.path.join(frame_path, city)
            frames = os.listdir(frame_file)
            masks = os.listdir(mask_file)
            frame_names = frame_names + frames
            for frame in frames:
                mask = frame.replace('leftImg8bit','gtFine_labelIds')
                frame_src = os.path.join(frame_file, frame)
                im = np.array(Image.open(frame_src))
                img = cv2.resize(im, None, fx=0.5,
                       fy=0.5, interpolation=cv2.INTER_LINEAR)
                x.append(img)
                
                if(split != 'test'): 
                    mask_src = os.path.join(mask_file, mask)
                    im = np.array(Image.open(mask_src))
                    mask = cv2.resize(im, None, fx=0.5,
                           fy=0.5, interpolation=cv2.INTER_NEAREST)  
                    y.append(mask)
                    
        if(name_list):
            print(len(frame_names), frame_names[0])
            return np.array(x), np.array(y), frame_names
        return np.array(x), np.array(y)


def save_results(files, result_dir, test_x, test_y, predict_y, split='test'):
    
    # map back to 0-33
    if(test_y.ndim > 2):
        test_y = np.argmax(test_y, axis=-1)
    predict_y = np.argmax(predict_y, axis=-1).astype(np.uint8)
    
    for i in range(len(files)):
        # 256,256,1 -- id --> change to color
        file_name = files[i][:-15]
        print(file_name)
        
        if(split != 'test'):
            label_gt = test_y[i]
            color_gt = palette(label_gt)
            imageio.imwrite(os.path.join(result_dir, file_name + 'B_color_gt.jpg'), color_gt.astype('uint8'))
            label_gt = scipy.misc.imresize(label_gt, (1024,2048),interp='nearest')
            np.save(os.path.join(result_dir, file_name + 'B_label_gt.npy'), label_gt)
            
        # if no test_y (gt) here
        label_pred = predict_y[i].astype('uint8')
        color_pred = palette(label_pred,(512,1024))
        label_pred = scipy.misc.imresize(label_pred, (1024,2048),interp='nearest')
        image = test_x[i]
        imageio.imwrite(os.path.join(result_dir, file_name + 'A.png'), image)
        imageio.imwrite(os.path.join(result_dir, file_name + 'B_labelId_pred.png'), label_pred)
        imageio.imwrite(os.path.join(result_dir, file_name + 'B_color_pred.png'), color_pred)
        
        
        

    
    
