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
from model import *
from newGen import *
from keras.utils import multi_gpu_model


def get_callbacks(name_weights, path, patience_lr, opt=1):
    mcp_save = ModelCheckpoint(name_weights, save_best_only=False, monitor='iou_score', mode='max')
    reduce_lr_loss = ReduceLROnPlateau(factor=0.5)
    logdir = os.path.join(path,'log')
    tensorboard = TensorBoard(log_dir=logdir, histogram_freq=0,
                                write_graph=True, write_images=True)
    if (opt == 3):
        return [mcp_save, tensorboard]
        
    else:
        return [mcp_save, reduce_lr_loss, tensorboard]

'''Options'''
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default='/home/yifan/Github/segmentation_train/dataset/')
parser.add_argument("--ckpt_path", type=str, default='/media/exfat/yifan/rf_checkpoints/cityscapes_unet_100e/')
parser.add_argument("--results_path", type=str, default='/media/exfat/yifan/rf_results/cityscapes_unet_100e/')
parser.add_argument("--network", type=str, default='Unet')
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--opt", type=int, default=1)
parser.add_argument("--n_classes", type=int, default=34)
parser.add_argument("--h", type=int, default=1024)
parser.add_argument("--w", type=int, default=2048)
parser.add_argument("--gpus", type=int, default=1)

args = parser.parse_args()

mkdir(args.ckpt_path)
with open(os.path.join(args.ckpt_path,'args.txt'), "w") as file:
    for arg in vars(args):
        print(arg, getattr(args, arg))
        file.write('%s: %s \n' % (str(arg),str(getattr(args, arg))))

BATCH_SIZE = args.batch_size
frame_path = os.path.join(args.dataset_path,'leftImg8bit')
mask_path = os.path.join(args.dataset_path,'gtFine')
cl = args.n_classes
h = args.h
w = args.w
gpus = args.gpus

'''define model
    
    Unet:  from segmentation_models
    unet & unet_noskip :  implemented in models module, warning when calculating the ERF
    
'''
input_shape = (h,w,3)
if (args.network == 'Unet'):
    m = Unet(classes = cl, input_shape=input_shape, activation='softmax')
#     m = get_unet()
elif (args.network == 'unet_noskip'):
    m = unet_noskip()
else:
    m = Unet('resnet18', classes=cl, input_shape=input_shape, activation='softmax')
m.summary()

#optimizer
if args.opt==1:
    opt= Adam(lr = 1e-4)
elif args.opt==2:
    opt = SGD(lr=0.01, decay=1e-6, momentum=0.99, nesterov=True)
else:
    opt = Adadelta(lr=1, rho=0.95, epsilon=1e-08, decay=0.0)
    
if (gpus >1 ):
    m = multi_gpu_model(m, gpus=gpus)

m.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=[iou_score])

# fit model
weights_path = args.ckpt_path + 'weights.{epoch:02d}-{val_loss:.2f}-{val_iou_score:.2f}.hdf5'
callbacks = get_callbacks(weights_path, args.ckpt_path, 5, args.opt)

train_generator,num_train = dataGen(frame_path, mask_path, BATCH_SIZE, args.epochs, (1024,2048))
val_generator,num_val = val_dataGen(frame_path, mask_path, 'val', BATCH_SIZE, args.epochs, (1024,2048))
history = m.fit_generator(
                        train_generator,
                        steps_per_epoch = num_train,
                        validation_data=val_generator,
                        validation_steps =num_val,
                        epochs=args.epochs,
                        verbose=1,
			            callbacks=callbacks
                        )

''' save model structure '''
model_json = m.to_json()
with open(os.path.join(args.ckpt_path,"model.json"), "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
print("Saved model to disk")
m.save(os.path.join(args.ckpt_path,'model.h5'))

'''Evaluate and Test '''
print('======Start Evaluating======')
test_generator,num_test = val_dataGen(frame_path, mask_path, 'test', BATCH_SIZE, args.epochs, (1024,2048))

score = m.evaluate_generatro(test_generator, steps=num_test)
print("%s: %.2f%%" % (m.metrics_names[0], score[0]*100))
print("%s: %.2f%%" % (m.metrics_names[1], score[1]*100))
with open(os.path.join(args.ckpt_path,'output.txt'), "w") as file:
    file.write("%s: %.2f%%" % (m.metrics_names[0], score[0]*100))
    file.write("%s: %.2f%%" % (m.metrics_names[1], score[1]*100))

print('======Start Testing======')
predict_y = m.predict_generator(test_generator, steps=num_test)

#save image
# print('======Save Results======')
# mkdir(args.results_path)
# save_results(mask_path, args.results_path, test_x, test_y, predict_y, 'test')



