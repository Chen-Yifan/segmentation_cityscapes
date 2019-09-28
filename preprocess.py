import os 
# from libtiff import TIFF
import numpy as np
from shutil import copyfile
from utils import *
import scipy.misc
from PIL import Image
import imageio

  
# Function to rename multiple files 
def clean_gtFine_tolabel(shape=256, cl=20): 
    '''delete non label files in gtFine'''
    directories = ['gtFine','leftImg8bit']
    splits = ['train','val','test']
    BASE_PATH = "/home/yifan/Github/segmentation_train/dataset/cityscapes_orig/"
    
    for d in directories:
        for split in splits:
            directory = os.path.join(BASE_PATH, d, split)
            cities = os.listdir(directory)
            
            for city in cities:
                ciy_dir = os.path.join(directory, city)
                all_files = os.listdir(ciy_dir)
                print(d,split,city,len(all_files), all_files[0])
                for file in all_files:
#                     print(file)
                    src = os.path.join(ciy_dir, file)
                    
                    # labelId to trainId and resize, save to file[:-12]+'trainIds.npy')
                    if(file[-12:] != 'labelIds.png'):
                        print('not',file)
                        os.remove(src)
                        continue
#                     '''If is labelId:
#                                 1. nearest neighbor resize to (shape,shape)
#                         '''
#                         im = np.array(Image.open(src))
#                         labelId = scipy.misc.imresize(im, (shape,shape),interp='nearest')
#                         dst = os.path.join(save_path, file[:-12]+'trainIds.npy') # *_labelIds.png
#                         np.save(dst,trainId)

def png2npy(shape=256, cl=20):
    directories = ['gtFine_label']
    splits = ['train','val','test']
    BASE_PATH = "/home/yifan/Github/segmentation_train/dataset/cityscapes_orig/png/"
    SAVE_PATH = "/home/yifan/Github/segmentation_train/dataset/cityscapes_orig/npy/"
    for d in directories:
        for split in splits:
            base_dir = os.path.join(BASE_PATH, d, split)
            save_dir = os.path.join(SAVE_PATH, d, split)
            mkdir(save_dir)
            
            all_files = os.listdir(base_dir)
            for file in all_files:
#                 print(file)
                src = os.path.join(base_dir, file)
                dst = os.path.join(save_dir, file[:-3]+'npy')
                print(src)
                print(dst)
            
                if(d == 'gtFine'):
                    '''If is labelId:
                        1. nearest neighbor resize to (shape,shape)
                    '''
                    im = np.array(Image.open(src))
                    labelId = scipy.misc.imresize(im, (shape,shape),interp='nearest')
                    
                    np.save(dst,labelId)
                
                else:
                    im = np.array(Image.open(src))
                    image = scipy.misc.imresize(im, (shape,shape),interp='bilinear')
                    np.save(dst,image)
                        
def argmax_20cl():
    directory = 'gtFine'
    splits = ['train','val','test']
    PATH = "/home/yifan/Github/segmentation_train/dataset/cityscapes_orig/"
    for split in splits:
        folder = os.path.join(PATH, directory, split)
        files = os.listdir(folder)
        for file in files:
            file_name = os.path.join(folder, file)
            image = np.argmax(np.load(file_name),axis=-1)
            print(file)
            np.save(file_name, image)

    
# Driver Code 
if __name__ == '__main__': 
      
    # Calling main() function 
    png2npy() 
