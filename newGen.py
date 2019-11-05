from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import random
from PIL import Image
import scipy.misc
import cv2
def zoom(image, label, crop_shape=(512,1024)):
    '''args:
            image: numpy array (original input 1024x2048
    '''
    (h,w) = crop_shape
    # random zoom
    zoom_factor = random.uniform(0.5,2)
    image = cv2.resize(image, None, fx=zoom_factor,
                       fy=zoom_factor, interpolation=cv2.INTER_LINEAR)/255.
    label = cv2.resize(label, None, fx=zoom_factor,
                       fy=zoom_factor, interpolation=cv2.INTER_NEAREST)
    #random crop
    top_crop = np.random.randint(0, image.shape[0] - h + 1)
    left_crop = np.random.randint(0, image.shape[1] - w + 1)
    image = image[top_crop: top_crop+h,
                  left_crop: left_crop+w]
    label = label[top_crop: top_crop+h,
                  left_crop: left_crop+w]
    return image , np.expand_dims(label,axis=-1)

def zoom_crop(gen, crop_size):
    '''random zoom between 0.5 to 2,
       random crop to crop_size
    '''
    while True:
        batch_x, batch_y = next(gen)
        batch_y = batch_y[:,:,:,0] # squeeze
        ret_x = np.zeros((batch_x.shape[0], crop_size[0], crop_size[1], 3))
        ret_y = np.zeros((batch_x.shape[0], crop_size[0], crop_size[1], 1))
        for i in range(batch_x.shape[0]):
            ret_x[i], ret_y[i] = zoom(batch_x[i], batch_y[i])
            # save image
            # print('save image %s'%i)
            # cv2.imwrite('./results/vis_augmentation/%s_input.png'%i, ret_x[i])
            # cv2.imwrite('./results/vis_augmentation/%s_input_orig.png'%i, batch_x[i])
        yield (ret_x, ret_y)
        
        
def dataGen(frame_path, mask_path, batch_size=1, shape=(1024,2048)):
    # Training path
    X_path= os.path.join(frame_path, 'train') # input image
    Y_path = os.path.join(mask_path, 'train') # ground-truth label

    # Train data generator
    x_gen_args = dict(
                    #rotation_range=0.2,
                    #width_shift_range=0.05,
                    #height_shift_range=0.05,
                    #shear_range=0.05,
                    horizontal_flip=True)

    y_gen_args = dict(
                    #rotation_range=0.2,
                    #width_shift_range=0.05,
                    #height_shift_range=0.05,
                    #shear_range=0.05,
                    horizontal_flip=True)

    img_datagen = ImageDataGenerator(**x_gen_args)
    mask_datagen = ImageDataGenerator(**y_gen_args)

    seed = 20 # the same seed is applied to both image_ and mask_generator
    image_generator = img_datagen.flow_from_directory(
        X_path,
        target_size=(1024,2048),
        batch_size=batch_size,
        shuffle = True, # shuffle the training data
        class_mode=None, # set to None, in this case
        interpolation='bilinear',
        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        Y_path,
        target_size=(1024,2048),
        color_mode='grayscale',
        batch_size=batch_size,
        shuffle = True,
        class_mode=None,
        interpolation='nearest',
        seed=seed)
    
    # combine image_ and mask_generator into one
    train_generator = zip(image_generator, mask_generator)
    train_generator = zoom_crop(train_generator,crop_size=shape)
    num_train = len(image_generator)
    
    return train_generator, num_train


# def val_dataGen(frame_path, mask_path, split, batch_size=1, shape=(1024,2048)):
#     h = shape[0] # image height
#     w = shape[1] # image width

#     # Validation path
#     val_X_path = os.path.join(frame_path, split)
#     val_Y_path = os.path.join(mask_path, split)

#     # val data generator
#     image_datagen = ImageDataGenerator(rescale = 1./255)
#     mask_datagen = ImageDataGenerator()
#     seed = 1
#     image_generator = image_datagen.flow_from_directory(
#         val_X_path,
#         target_size=shape,
#         batch_size=batch_size,
#         class_mode=None,
#         interpolation='bilinear',
#         shuffle = False, # we dont need to shuffle validation set
#         seed=seed)

#     mask_generator = mask_datagen.flow_from_directory(
#         val_Y_path,
#         target_size=shape,
#         color_mode='grayscale',
#         batch_size=batch_size,
#         shuffle = False,
#         class_mode=None,
#         interpolation='nearest',
#         seed=seed)

#     val_generator = zip(image_generator, mask_generator)
# #     val_resize = crop_generator(val_generator, (h,w), random=False)
#     num_val = len(image_generator)
#     return val_generator, num_val
