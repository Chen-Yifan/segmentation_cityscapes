import os 
# from libtiff import TIFF
import numpy as np
  
# Function to rename multiple files 
def main(): 
    directories = ['gtFine', 'leftImg8bit']
    splits = ['train', 'val', 'test']
    BASE_PATH = "/home/yifan/Github/segmentation_train/dataset/"
    NEW_PATH = "/home/yifan/Github/segmentation_train/dataset/cityscapes_all/"
    
    for d in directories:
        for split in splits:
            directory = os.path.join(BASE_PATH, d, split)
            save_path = os.path.join(NEW_PATH, d, split) # .../dataset/cityscapes_all/gtFine/train
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            cities = os.listdir(directory)
            i = 0
            for city in cities:
                cur_dir = os.path.join(directory, city)
                all_files = os.listdir(cur_dir)
#                 print(d,split,city,len(all_files), all_files[0])
                for file in all_files:
#                     if (d == 'gtFile' and file[-12:] != 'labelIds.png'):
#                         print('not', file)
#                         continue
                    print(file)
                    src = os.path.join(cur_dir, file)
                    dst = os.path.join(save_path, file)
                    print('src:', src)
                    print('dst:', dst)
                    os.rename(src, dst)
                    i+=1

                    
def mv_frames():
    PATH = "/home/yifanc3/dataset/data/"
    NEW_FRAME = PATH + 'masks_5m_256overlap'
    SAVE_FRAME = PATH + 'selected_256_overlap/all_masks_5m'
    if not os.path.isdir(SAVE_FRAME):
        os.makedirs(SAVE_FRAME)
        
    FRAME_PATH = PATH + 'selected_256_overlap/all_masks_10m'
    all_frames = os.listdir(FRAME_PATH)
    for frame in all_frames:
        src = os.path.join(NEW_FRAME, frame[:-3]+'tif')
        tif = TIFF.open(src)
        img = tif.read_image() # 0 means annotated
        dst = os.path.join(SAVE_FRAME, frame)
        print(src, dst)
        np.save(dst,img)

    
# Driver Code 
if __name__ == '__main__': 
      
    # Calling main() function 
    main() 
#     mv_frames()

