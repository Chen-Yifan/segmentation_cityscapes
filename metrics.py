import keras.backend as K
import numpy as np
import os
import glob
import skimage.io as io
import tensorflow as tf
from segmentation_models.metrics import iou_score
labels = __import__('labels')
Id2ignore = {label.id: label.ignoreInEval for label in labels.labels} 

def Mean_IOU(y_true, y_pred):
    nb_classes = K.int_shape(y_pred)[-1] 
    print(nb_classes)
    iou = []
    true_pixels = K.squeeze(y_true,axis=-1)  # = (n, h,w)
    pred_pixels = K.argmax(y_pred, axis=-1) 
    #ignore certain labels, those doesn't have one, and those are background
    void_labels = K.equal(true_pixels, 19) # ignore label 19, background

    # in our case, the last label is background (19)
    for i in range(0, nb_classes): # exclude first label (background) and last label (void)
        true_labels = K.equal(true_pixels, i) & ~void_labels
        pred_labels = K.equal(pred_pixels, i) & ~void_labels
        inter = tf.to_int32(true_labels & pred_labels)
        union = tf.to_int32(true_labels | pred_labels)
        legal_batches = K.sum(tf.to_int32(true_labels), axis=1)>0
        ious = K.sum(inter, axis=1)/K.sum(union, axis=1)
        iou.append(K.mean(tf.gather(ious, indices=tf.where(legal_batches)))) # returns average IoU of the same objects
    iou = tf.stack(iou)
    legal_labels = ~tf.debugging.is_nan(iou)
    iou = tf.gather(iou, indices=tf.where(legal_labels))
    return K.mean(iou)

def meaniou(y_true, y_pred):
    '''
    In this function, we calculate the meaniou between labels and predictions 
    y_true is in (batch, h, w, 1), we need to cast it into 'int8' and reshape to (batch, h, w, cl)
    after the transformation, we calculate the meaniou
    '''
    y_true = K.squeeze(y_true,axis=-1)  
    y_true = K.cast(y_true, 'int32')
    cl = K.int_shape(y_pred)[-1] 
    y_true = K.one_hot( y_true, num_classes=cl)
    return iou_score(y_true, y_pred)

def meaniou_partial(y_true, y_pred):
    '''this function maintains most of the functionality in meaniou,
    Only that we ignore those ignoreInEval classess listed in labels
    '''
    y_true = K.squeeze(y_true,axis=-1)  
    y_true = K.cast(y_true, 'int32')
    shape = K.int_shape(y_pred)
    cl = shape[-1] 
    y_true = K.one_hot( y_true, num_classes=cl)
    
    
    n = 0
    id_list = []
    for i in range(cl): #0...33
        if(Id2ignore[i] == False): # not ignore
            n += 1
            id_list.append(i)
    print('n is' ,n)
    
#     y_pred = np.array(y_pred)
#     y_true = np.array(y_true)
#     print(y_pred.shape, y_true.shape)
#     y_pred_clean = np.concatenate([y_pred[:,:,:,a] for a in id_list], axis=-1)
#     y_true_clean = np.concatenate([y_true[:,:,:,a] for a in id_list], axis=-1)
    
    return iou_score(y_true_clean, y_pred_clean) 
            
