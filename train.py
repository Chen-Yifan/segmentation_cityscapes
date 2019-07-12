from segmentation_models import Unet
# from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau


def get_callbacks(name_weights, path, patience_lr):
    mcp_save = ModelCheckpoint(name_weights, save_best_only=False, monitor='iou_score', mode='max')
#     reduce_lr_loss = ReduceLROnPlateau(monitor='bce_jaccard_loss', factor=0.5, patience=patience_lr, verbose=1, epsilon=1e-4, mode='min')
    logdir = os.path.join(path,'log')
    tensorboard = TensorBoard(log_dir=logdir, histogram_freq=0,
                            write_graph=True, write_images=True)
    return [mcp_save, tensorboard]


import os
import argparse
from dataGenerator import *

#get arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default='/home/yifan/Github/segmentation_train/dataset/cityscapes')
parser.add_argument("--ckpt_path", type=str, default='./models/')
parser.add_argument("--results_path", type=str, default='')
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--epochs", type=int, default=100)
args = parser.parse_args()

BATCH_SIZE = args.batch_size

if not os.path.isdir(args.ckpt_path):
    os.makedirs(args.ckpt_path)

#data path
folder = ['trainB','trainA', 'testB', 'testA']
train_mask_path = os.path.join(args.dataset_path, folder[0])
train_frame_path = os.path.join(args.dataset_path, folder[1])
test_mask_path = os.path.join(args.dataset_path, folder[2])
test_frame_path =  os.path.join(args.dataset_path, folder[3])


NO_OF_TRAINING_IMAGES = int(0.7*len(os.listdir(train_mask_path)))
NO_OF_VAL_IMAGES = len(os.listdir(train_mask_path)) - NO_OF_TRAINING_IMAGES
print(NO_OF_TRAINING_IMAGES, NO_OF_VAL_IMAGES)

# load your data
train_x, train_y, val_x, val_y = load_trainval(train_mask_path, train_frame_path, BATCH_SIZE)
train_gen, val_gen = dataGen(train_x, train_y, val_x, val_y, BATCH_SIZE)

# define model
m = Unet(classes = 20, input_shape=(256, 256, 3))
m.summary()
#optimizer
opt = 'Adam'
m.compile(opt, loss=bce_jaccard_loss, metrics=[iou_score])

# fit model
# if you use data generator use model.fit_generator(...) instead of model.fit(...)
# more about `fit_generator` here: https://keras.io/models/sequential/#fit_generator
weights_path = args.ckpt_path + 'weights.{epoch:02d}-{val_loss:.2f}-{val_Mean_IOU:.2f}.hdf5'
    
callbacks = get_callbacks(weights_path, args.ckpt_path, 5)

history = m.fit_generator(train_gen, epochs=args.epochs,
                          steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                          validation_data=val_gen,
                          validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE),
                          shuffle = True,
                          callbacks=callbacks)


test_x, test_y = load_test(test_frame_path, test_mask_path)
results = m.predict(test_x)

#save image
save_result(args.results_path, results, test_x, test_y)



