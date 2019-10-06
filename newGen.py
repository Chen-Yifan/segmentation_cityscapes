from keras.preprocessing.image import ImageDataGenerator
import os

def dataGen(batch_size=1, epochs=1, shape=256):
    # Training path
    frame_path = '/home/yifan/Github/segmentation_train/dataset/cityscapes_orig/png/leftImg8bit'
    mask_path = '/home/yifan/Github/segmentation_train/dataset/cityscapes_orig/png/gtFine_label'
    X_path= os.path.join(frame_path, 'train') # input image
    Y_path = os.path.join(mask_path, 'train') # ground-truth label

    h = shape # image height
    w = shape # image width

    # Validation path
    val_X_path = os.path.join(frame_path, 'val')
    val_Y_path = os.path.join(mask_path, 'val')

    # Train data generator
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

    seed = 1 # the same seed is applied to both image_ and mask_generator
    image_generator = img_datagen.flow_from_directory(
        X_path,
        target_size=(h, w),
        batch_size=batch_size,
        shuffle = True, # shuffle the training data
        class_mode=None, # set to None, in this case
        interpolation='nearest',
        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        Y_path,
        target_size=(h, w),
        color_mode='grayscale',
        batch_size=batch_size,
        shuffle = True,
        class_mode=None,
        interpolation='nearest',
        seed=seed)
    
    # combine image_ and mask_generator into one
    train_generator = zip(image_generator, mask_generator)
    num_train = len(image_generator)
    
    # val data generator
    image_datagen = ImageDataGenerator(rescale = 1./255)
    mask_datagen = ImageDataGenerator()
    seed = 1
    image_generator = image_datagen.flow_from_directory(
        val_X_path,
        target_size=(h, w),
        batch_size=batch_size,
        class_mode=None,
        shuffle = False, # we dont need to shuffle validation set
        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        val_Y_path,
        target_size=(h, w),
        color_mode='grayscale',
        batch_size=batch_size,
        shuffle = False,
        class_mode=None,
        seed=seed)

    val_generator = zip(image_generator, mask_generator)
    num_val = len(image_generator)
    
    print(num_train, num_val)
    return train_generator,val_generator, num_train, num_val
