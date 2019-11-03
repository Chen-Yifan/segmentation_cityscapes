from utils import *
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
import os
import tensorflow as tf
import keras.backend as K
import numpy as np
from keras.optimizers import Adadelta, Adam
import matplotlib.pyplot as plt
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from keras.optimizers import SGD,Adam,Adadelta
from dataGenerator import *
from keras.models import model_from_json
from segmentation_models import Unet
import argparse
from keras.utils import multi_gpu_model
from metrics import *
from newGen import *

#get arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default='/home/yifan/Github/segmentation_train/dataset/cityscapes_all')
parser.add_argument("--ckpt_path", type=str, default='/media/exfat/yifan/rf_checkpoints/cityscapes_unet_100e/')
parser.add_argument("--results_path", type=str, default='/media/exfat/yifan/rf_results/cityscapes_unet_100e/')
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--opt", type=str, default='SGD')
parser.add_argument("--h", type=int, default=1024)
parser.add_argument("--w", type=int, default=2048)
parser.add_argument("--split", type=str, default='test')
parser.add_argument("--gpus", type=int, default=2)
args = parser.parse_args()
h = args.h
w = args.w

gpus = args.gpus
BATCH_SIZE = args.batch_size
frame_path = os.path.join(args.dataset_path,'leftImg8bit')
mask_path = os.path.join(args.dataset_path,'gtFine')

test_x, test_y, test_files = load_test(mask_path, frame_path, args.split, True, (h,w), True)
#print(test_x.shape)
#assert(len(test_files)==len(test_x))
#print(test_files[0])

#9-11 the epoch
weights = os.listdir(args.ckpt_path)
weight = None
for i in weights: 
    if i[8:10] == str(args.epochs):
        weight = i
print(weight)
Model_dir = os.path.join(args.ckpt_path,weight)

#model 

json_path = os.path.join(args.ckpt_path,'model.json')
json_file = open(json_path, 'r')
loaded_model_json = json_file.read()
json_file.close()

m = model_from_json(loaded_model_json)
m.load_weights(Model_dir)

#optimizer
if args.opt=='Adam':
    opt= Adam(lr = 1e-4)
elif args.opt=='SGD':
    opt = SGD(lr=0.01, decay=5e-4, momentum=0.90)
else:
    opt = Adadelta(lr=1, rho=0.95, epsilon=1e-08, decay=0.0)

#m = multi_gpu_model(m, gpus=2)
#print("Training using multiple GPUs..")
        
def sparse_softmax_cce(y_true, y_pred):
    y_true = K.squeeze(y_true, axis=-1)
    y_true = tf.cast(y_true, 'int32')
    print(y_true.get_shape())
    return tf.keras.backend.sparse_categorical_crossentropy(y_true,y_pred)

m.compile(optimizer=opt, loss=sparse_softmax_cce, metrics=[meaniou])

if (args.split=='val'):
    score = m.evaluate(test_x/255, test_y, verbose=0)
    print("%s: %.2f%%" % (m.metrics_names[0], score[0]*100))
    print("%s: %.2f%%" % (m.metrics_names[1], score[1]*100))
    with open(os.path.join(args.ckpt_path,'output%s.txt'% args.epochs), "w") as file:
        file.write("%s: %.2f%%" % (m.metrics_names[0], score[0]*100))
        file.write("%s: %.2f%%" % (m.metrics_names[1], score[1]*100))

#test_generator,num_test = val_dataGen(frame_path, mask_path, 'test', 1, (h, w))
predict_y = m.predict(test_x/255)
#predict_y = m.predict_generator(test_generator, steps=num_test, verbose=0)

result_path = args.results_path +'weights.%s-results-%s'%(args.epochs, args.split)
print(result_path)
mkdir(result_path)

#save image
save_results(test_files, result_path, test_x, test_y, predict_y, split=args.split)
