import numpy as np
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import itertools

import NormalizeStaining
import RollingWindow

#pre-processing
def myFunc(e):
  return e[0]

#this func finds max ratio in 512x512 pixel images. (sütün, satir)
# def find_patches_from_image(list_tensor,patch_size):
# #  cell_label = count = 0
#   csv_file = []
#   start_sütün = 0
#   start_satir = 0
#   for index in range(len(list_tensor)):
#      ##########burada kaldım 
#     patch_ratio = get_patch_ratio(list_tensor[index],patch_size)
#     img_arr = []
#     img_arr.append(patch_ratio)
#     img_arr.append(list_tensor[index])
#     img_arr.append(start_sütün)
#     img_arr.append(start_satir)

#     csv_file.append(img_arr)

#     start_satir = start_satir +512
#     if start_satir>=2048 :
#       start_satir=0
#       start_sütün = start_sütün + 512


#   csv_file.sort(reverse=True, key=myFunc)
  
  # return csv_file[0][0],csv_file[0][1],csv_file[0][2],csv_file[0][3]

def get_patch_ratio(patch:np.ndarray, patch_size:int,cell_label = 0 ):
   return np.count_nonzero(patch == cell_label) / (patch_size ** 2)

def get_patch_ratio_for_image(patch:np.ndarray,cell_label = 0 ):
   return np.count_nonzero(patch == cell_label) / (1536*2048)

def find_patches_from_patch(list_tensorr,patch_size,step):
#  cell_label = 0
  csv_file = []

  for index in range(len(list_tensorr)):
    patch_ratio = get_patch_ratio(list_tensorr[index][0],patch_size)
    img_arr = []
    if ( patch_ratio>=0.35 and patch_ratio<=0.65):
      img_arr.append(patch_ratio)
      img_arr.append(list_tensorr[index][1])
      img_arr.append(list_tensorr[index][2])
     
      csv_file.append(img_arr)
      
  csv_file.sort(reverse=True, key=myFunc)
  return csv_file



#pre-processing (Custom Funcs) Patch
def patch_images(glob_variables, image_data_dir, patch_data_dir):
    class_labels= ['Benign','InSitu','Invasive','Normal']
    class_short_label = ['Bn','Is','Iv','Nr']
#    savefile =  'train/'
    patch_size = glob_variables['img_width']
#    classNo = 0 
#    images = []
#    labels = []

#    count = 0
    step = 8
#    ratios = []
    
    print("patch_images called for " + image_data_dir)
    for index, fold_dir in enumerate(class_labels):
        print(fold_dir)
        
        train_tiffs = []
#        train_tiffs = sorted(os.listdir(glob_variables['bach_data_dir']+fold_dir+'/'))
#        train_tiffs = sorted(os.listdir(glob_variables['bioimaging_data_dir']+fold_dir+'/'))
        train_tiffs = sorted(os.listdir(image_data_dir+fold_dir+'/'))
        print(train_tiffs)

        save_image_size = 1
        for image_dir in train_tiffs:
            
            
            directory = image_data_dir + class_labels[index] + '/' + image_dir
            savedir = patch_data_dir +  class_labels[index] + '/' + class_short_label[index]
#            print(directory)
            img = cv2.imread(directory)
#            plt.imshow(img)
            print(directory)
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            normal_image = NormalizeStaining.normalizeStaining(image)
            gray = cv2.cvtColor(normal_image, cv2.COLOR_BGR2GRAY)
            (thresh, blackAndWhiteImage) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            
            #imageRatio= get_patch_ratio_for_image(blackAndWhiteImage)
            #ratios.append(imageRatio)
            list_patched = RollingWindow.rolling_window(blackAndWhiteImage, patch_size, step, print_dims = False)
    
            list_images = find_patches_from_patch(list_patched,patch_size,step)
    
            img = np.array(image)
            if ( len( list_images ) > glob_variables['patch_image_count'] ):
              for indis in range( glob_variables['patch_image_count'] ):
                start_row = list_images[indis][1]
                start_col = list_images[indis][2]
                temp= img[start_row:start_row+patch_size ,start_col:start_col+patch_size]
                saveimagedir = savedir + str(save_image_size)
                Image.fromarray(temp).save(saveimagedir+'.png')
                save_image_size +=1
                
            else:
              for indis in range( len(list_images) ):
                start_row = list_images[indis][1]
                start_col = list_images[indis][2]
                temp= img[start_row:start_row+patch_size ,start_col:start_col+patch_size]
                saveimagedir = savedir + str(save_image_size)
                Image.fromarray(temp).save(saveimagedir+'.png')
                save_image_size +=1




def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.close()
    plt.imshow(cm, interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # plt.show()
    return plt.savefig("confusion_matrix.png")
    # plt.tight_layout()

def Average(lst): 
    return sum(lst) / len(lst)


# conda install keras=2.0.5