from segmentation_models import Unet
# from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from keras.optimizers import SGD,Adam,Adadelta
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from dataGenerator import *
from utils import *
from metrics import *
import os
import argparse


def get_callbacks(name_weights, path, patience_lr):
    mcp_save = ModelCheckpoint(name_weights, save_best_only=False, monitor='iou_score', mode='max')
#     reduce_lr_loss = ReduceLROnPlateau(monitor='bce_jaccard_loss', factor=0.5, patience=patience_lr, verbose=1, epsilon=1e-4, mode='min')
    # reduce_lr_loss = ReduceLROnPlateau(factor=0.5)
    logdir = os.path.join(path,'log')
    tensorboard = TensorBoard(log_dir=logdir, histogram_freq=0,
                            write_graph=True, write_images=True)
    return [mcp_save, tensorboard]

#get arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default='/home/yifan/Github/segmentation_train/dataset/cityscapes_all')
parser.add_argument("--ckpt_path", type=str, default='/media/exfat/yifan/rf_checkpoints/cityscapes_unet_100e/')
parser.add_argument("--results_path", type=str, default='/media/exfat/yifan/rf_results/cityscapes_unet_100e/')
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=100)

args = parser.parse_args()

mkdir(args.ckpt_path)
with open(os.path.join(args.ckpt_path,'args.txt'), "w") as file:
    for arg in vars(args):
        print(arg, getattr(args, arg))
        file.write('%s: %s \n' % (str(arg),str(getattr(args, arg))))

BATCH_SIZE = args.batch_size
frame_path = os.path.join(args.dataset_path,'left_256')
mask_path = os.path.join(args.dataset_path,'gtFine_256')


#write to output

# load data to lists
train_x, train_y, val_x, val_y, test_x, test_y = load_data(frame_path, mask_path)
print('train_y.shape:',train_y.shape)

NO_OF_TRAINING_IMAGES = train_x.shape[0]
NO_OF_VAL_IMAGES = val_x.shape[0]
print('train: val', NO_OF_TRAINING_IMAGES, NO_OF_VAL_IMAGES)

#DATA AUGMENTATION
train_gen, val_gen = dataGen(train_x, train_y, val_x, val_y, BATCH_SIZE)

# define model
m = Unet(classes = 20, input_shape=(256, 256, 3), activation='softmax')
m.summary()

#optimizer
opt= Adam(lr = 1e-4)
opt2 = SGD(lr=0.01, decay=1e-6, momentum=0.99, nesterov=True)
opt3 = Adadelta(lr=1, rho=0.95, epsilon=1e-08, decay=0.0)
m.compile(optimizer=opt3, loss='categorical_crossentropy', metrics=[iou_score])

# fit model
weights_path = args.ckpt_path + 'weights.{epoch:02d}-{val_loss:.2f}-{val_iou_score:.2f}.hdf5'
callbacks = get_callbacks(weights_path, args.ckpt_path, 5)
history = m.fit_generator(train_gen, epochs=args.epochs,
                          steps_per_epoch = (NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                          validation_data=val_gen,
                          validation_steps=(NO_OF_VAL_IMAGES//BATCH_SIZE),
                          shuffle = True,
                          callbacks=callbacks)
#save model structure
model_json = m.to_json()
with open(os.path.join(args.ckpt_path,"model.json"), "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
print("Saved model to disk")
m.save(os.path.join(args.ckpt_path,'model.h5'))

#prediction
print('======Start Evaluating======')
score = m.evaluate(test_x, test_y, verbose=0)
print("%s: %.2f%%" % (m.metrics_names[1], score[1]*100))
with open(os.path.join(args.ckpt_path,'output.txt'), "w") as file:
    file.write("%s: %.2f%%" % (m.metrics_names[1], score[1]*100))

print('======Start Testing======')
predict_y = m.predict(test_x)

#save image
print('======Save Results======')
mkdir(args.results_path)
save_results(mask_path, args.results_path, test_x, test_y, predict_y)



