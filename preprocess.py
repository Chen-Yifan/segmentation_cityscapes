import os
import re 
# from libtiff import TIFF
import numpy as np
from shutil import copyfile
from utils import *
import scipy.misc
from PIL import Image
import imageio
from dataGenerator import labelId2trainId

  
# Function to rename multiple files 
def main(shape=256, cl=20): 
    directories = ['leftImg8bit']
    splits = ['train','val']
    BASE_PATH = "/home/yifanc3/Github/segmentation_train/dataset/"
    NEW_PATH = "/home/yifanc3/Github/segmentation_train/dataset/cityscapes_256/"
    
    for d in directories:
        for split in splits:
            directory = os.path.join(BASE_PATH, d, split)
            save_path = os.path.join(NEW_PATH, d, split) # .../dataset/cityscapes_all/gtFine/train
            mkdir(save_path)
            cities = os.listdir(directory)
            i = 0
            for city in cities:
                cur_dir = os.path.join(directory, city)
                all_files = os.listdir(cur_dir)
                print(d,split,city,len(all_files), all_files[0])
                for file in all_files:
                    print(file)
                    src = os.path.join(cur_dir, file)
                    
                    # read image and resize to 256, 256, save as npy
                    if (d == 'leftImg8bit'):
                        # read image
                        im = np.array(Image.open(src))
                        # resize
                        frame = scipy.misc.imresize(im, (shape, shape), interp='bilinear')
                        dst = os.path.join(save_path, file[:-3]+'npy')
                        # *_leftImg8bit.png
                        np.save(dst,frame)
                    
                    # labelId to trainId and resize, save to file[:-12]+'trainIds.npy')
                    if (d == 'gtFine'):
                        if(file[-12:] != 'labelIds.png'):
                            print('not',file)
                            continue
                        '''If is labelId:
                                1. nearest neighbor resize to (shape,shape)
                                2. change to trainId (255 to 19)
                                3. one hot encode and save
                        '''
                        im = np.array(Image.open(src))
                        labelId = scipy.misc.imresize(im, (shape,shape),interp='nearest')
                        trainId = labelId2trainId(labelId)
#                         mask = np.eye(cl)[trainId]
                        dst = os.path.join(save_path, file[:-12]+'trainIds.npy') # *_labelIds.png
                        np.save(dst,trainId)
                    i+=1

def gen_new_test():
    PATH = "/home/yifanc3/Github/segmentation_train/dataset/cityscapes_256/gtFine_256/"
    OLD_PATH = PATH + 'train'
    NEW_PATH = PATH + 'test'
    mkdir(NEW_PATH)
    all_files = os.listdir(OLD_PATH)
    all_files.sort(key=lambda var:[int(x) if x.isdigit() else x 
                                for x in re.findall(r'[^0-9]|[0-9]+', var)])   
    all_files = all_files[0:699]
    print(len(all_files))
    for file in all_files:
        src = os.path.join(OLD_PATH, file)
        dst = os.path.join(NEW_PATH, file)
        os.rename(src,dst)
        print(dst)
    
# Driver Code 
if __name__ == '__main__': 
      
    ## Calling main() function 
   # main()
    gen_new_test() 

