from keras.preprocessing.image import ImageDataGenerator

batch_size = 1
epochs = 50
h = 360 # image height
w = 480 # image width

# Training path
X_path= os.path.join('camvid', 'train') # input image
Y_path = os.path.join('camvid', 'trainannot') # ground-truth label

# Validation path
val_X_path = os.path.join('camvid', 'val')
val_Y_path = os.path.join('camvid', 'valannot')

# Train data generator
x_gen_args = dict(
                        rescale=1./255,
                        #featurewise_center=True,
                        #featurewise_std_normalization=True,
                        #shear_range=0.2,
                        #zoom_range=0.5,
                        #channel_shift_range=?,
                        #width_shift_range=0.5,
                        #height_shift_range=0.5,
                        rotation_range = 10,
                        horizontal_flip=True
                    )
y_gen_args = dict(
                        #featurewise_center=True,
                        #featurewise_std_normalization=True,
                        #shear_range=0.2,
                        #zoom_range=0.5,
                        #channel_shift_range=?,
                        #width_shift_range=0.5,
                        #height_shift_range=0.5,
                        rotation_range = 10,
                        horizontal_flip=True
                    )

image_datagen = ImageDataGenerator(**x_gen_args)
mask_datagen = ImageDataGenerator(**y_gen_args)

seed = 1 # the same seed is applied to both image_ and mask_generator
image_generator = image_datagen.flow_from_directory(
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
image_datagen = ImageDataGenerator()
mask_datagen = ImageDataGenerator()
seed = 1
image_generator = image_datagen.flow_from_directory(
    val_X_path,
    target_size=(ch, cw),
    batch_size=batch_size,
    shuffle = False, # we dont need to shuffle validation set
    class_mode=None,
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    val_Y_path,
    target_size=(ch, cw),
    color_mode='grayscale',
    batch_size=batch_size,
    shuffle = False,
    seed=seed)

val_generator = zip(image_generator, mask_generator)
num_val = len(image_generator)

# fit the generators
model.fit_generator(
                    train_generator,
                    steps_per_epoch = num_train/batch_size, 
                    validation_data=val_generator,
                    validation_steps =num_val/batch_size,
                    epochs=epochs,
                    verbose=1
                    )