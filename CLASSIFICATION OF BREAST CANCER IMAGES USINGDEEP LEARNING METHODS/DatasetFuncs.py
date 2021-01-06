import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.applications.nasnet import NASNetLarge
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.densenet import DenseNet201
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet import ResNet152
from keras.applications.inception_v3 import InceptionV3
from keras.utils.np_utils import to_categorical
from keras.applications.resnet_v2 import ResNet152V2
from keras.applications.mobilenet_v2 import MobileNetV2
import matplotlib.pyplot as plt
import math


#Save BACH Dataset
def save_bach_bottleneck(glob_variables):
    model = InceptionResNetV2(include_top=False, weights='imagenet')

    # datagen = ImageDataGenerator(rescale=1. / 255, 
    #                              rotation_range=90)
    datagen = ImageDataGenerator(rescale=1. / 255)
  
    generator = datagen.flow_from_directory(
        glob_variables['bach_patch_images_data_dir'] + str(glob_variables['img_width']),
        target_size=(glob_variables['img_width'], 
        glob_variables['img_height']),
        batch_size=glob_variables['batch_size'],
        class_mode=None,
        shuffle=False)

    nb_validation_samples = len(generator.filenames)
    print("nb_validation_samples", nb_validation_samples)
  
    predict_size_validation = int(math.ceil(nb_validation_samples / glob_variables['batch_size']))
    print("predict_size_validation", predict_size_validation)
  
    bottleneck_features_validation = model.predict(generator, 
                                                 steps = predict_size_validation)
  
    np.save('bach/bottleneck_features_train_bach_' + glob_variables['model'] +'_' + str(glob_variables['img_width']) +'.npy',
            bottleneck_features_validation)

  
def load_bach_bottleneck(glob_variables):  
    print('bach/bottleneck_features_train_bach_' + glob_variables['model'] +'_' + str(glob_variables['img_width']) +'.npy')
    datagen_top = ImageDataGenerator(rescale=1. / 255)
    num_classes = 4
    generator_top = datagen_top.flow_from_directory(
        glob_variables['bach_patch_images_data_dir'] + str(glob_variables['img_width']),
        target_size=(glob_variables['img_width'], glob_variables['img_height']),
        batch_size=glob_variables['batch_size'],
        class_mode=None,
        shuffle=False)

#  nb_validation_samples = len(generator_top.filenames)

    validation_data = np.load('bach/bottleneck_features_train_bach_' + glob_variables['model'] +'_' + str(glob_variables['img_width']) +'.npy')

    validation_labels = generator_top.classes
    # print(validation_labels.shape)
    validation_labels = to_categorical(validation_labels, 
                                     num_classes=num_classes)
  
    print(validation_data.shape)
    print(validation_labels.shape)
    return validation_data,validation_labels



#Save Bioimaging Dataset
def save_bioimaging_bottleneck(glob_variables):
    model = InceptionResNetV2(include_top=False, weights='imagenet')
    
    datagen = ImageDataGenerator(rescale=1. / 255)

    generator = datagen.flow_from_directory(glob_variables['bioimaging_patch_images_data_dir'] + str(glob_variables['img_width']),
                                            target_size=(glob_variables['img_width'], glob_variables['img_height']),
                                            batch_size=glob_variables['batch_size'],
                                            class_mode=None,
                                            shuffle=False)

    nb_validation_samples = len(generator.filenames)

    predict_size_validation = int(
        math.ceil(nb_validation_samples / glob_variables['batch_size']))
    
    bottleneck_features_validation = model.predict(
        generator, predict_size_validation)
    
    np.save('bioimaging/bottleneck_features_validation_bioimaging_' + glob_variables['model'] + '_' + str(glob_variables['img_width']) + '.npy',
            bottleneck_features_validation)
    
    
#Load Bioimaging Dataset

def load_bioimaging_bottleneck(glob_variables):
    print('bioimaging/bottleneck_features_validation_bioimaging_' + glob_variables['model'] + '_' + str(glob_variables['img_width']) + '.npy')
    datagen = ImageDataGenerator(rescale=1. / 255)
    num_classes = 4
    generator_top = datagen.flow_from_directory(
        glob_variables['bioimaging_patch_images_data_dir'] + str(glob_variables['img_width']),                    
        target_size=(glob_variables['img_width'], glob_variables['img_height']),
        batch_size=glob_variables['batch_size'],
        class_mode=None,
        shuffle=False)

#    nb_validation_samples = len(generator_top.filenames)

    validation_data = np.load('bioimaging/bottleneck_features_validation_bioimaging_' + glob_variables['model'] + '_' + str(glob_variables['img_width']) + '.npy')

    validation_labels = generator_top.classes
    validation_labels = to_categorical(
        validation_labels, num_classes=num_classes)

    return validation_data,validation_labels




# 'bach/bottleneck_features_train_bach_InceptionResNetV2.npy'                     2.4gb
# 'bioimaging/bottleneck_features_validation_bioimaging_InceptionResNetV2.npy'    1.4gb

# 'bach/bottleneck_features_train_bach_Xception.npy'                              5.8gb
# 'bioimaging/bottleneck_features_validation_bioimaging_Xception.npy'             3.4gb

# 'bach/bottleneck_features_train_bach_NASNetLarge.npy'                           11.5gb
# 'bioimaging/bottleneck_features_validation_bioimaging_NASNetLarge.npy'          6.7gb



# python -m PyQt5.uic.pyuic -x final.ui -o final.py

