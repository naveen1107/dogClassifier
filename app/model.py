from extract_bottleneck_features import *
from keras.models import load_model
#from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

import cv2                
#import matplotlib.pyplot as plt  

from extract_bottleneck_features import *

from keras.callbacks import ModelCheckpoint 
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dropout, Dense
from keras.models import Sequential
from keras.applications.xception import Xception, preprocess_input
import numpy as np
import tensorflow as tf

from detector import get_dog_names, dog_detector, face_detector
dog_names = get_dog_names()

from keras import backend as K

from preprocess import path_to_tensor


 
def predict_dog_breed_with_img(img_path, model = None, model_path = None, feature_type = 'VGG16'):
    
    """
        Function to predict dog breed
        
        Args;
            img_path: path to image
            model: reference to model
            model_path: path to saved model
            feature_type: type of bottlenck feature
        
        returns:
            return the name of dog breed classified
    """
    # load model
    if model is None:
        if model_path is not None:
            model = load_model(model_path)
        else:
            raise ValueError('Please pass either model reference of model_path')
    
    # extract features
    if feature_type == 'VGG16':
        features = extract_VGG16(path_to_tensor(img_path))
    elif feature_type == 'VGG19':
        features = extract_VGG19(path_to_tensor(img_path))
    elif feature_type == 'Resnet50':
        #features = preprocess_input(path_to_tensor(img_path))
        
        #features = extract_Resnet50(image_preprocess(img_path))
        features = extract_Resnet50(path_to_tensor(img_path))
    elif feature_type == 'Xception':
        features = extract_Xception(path_to_tensor(img_path))
    elif feature_type == 'InceptionV3':
        features = extract_InceptionV3(path_to_tensor(img_path))
    else:
        raise ValueError('Please Pass correct features type:\n \{ VGG16, VGG19, Resnet50, Xception, InceptionV3 \} ')
    
    # obtain predicted vector
    predicted_vector = model.predict(features)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]
    
def load_trained_model():

    K.clear_session()
    model_path = './weights.best.Resnet50.hdf5'
    feature_type = 'Resnet50'
    #model = load_model(model_path)
    #dog_model = ResNet50(weights='imagenet')
    
    Resnet50_model = Sequential()
    Resnet50_model.add(GlobalAveragePooling2D(input_shape=(1, 1, 2048)))
    Resnet50_model.add(Dense(256, activation = 'relu'))
    Resnet50_model.add(Dense(133, activation='softmax'))
    
    
    #Xception_model.load_weights(model_path)
    Resnet50_model.load_weights(model_path)
    model = Resnet50_model
    print(model.summary())
    return model, feature_type


def my_dog_detector(img_path = ''):
    
    model, feature_type = load_trained_model()
    
    print(feature_type)
    
    if dog_detector(img_path):
        breed = predict_dog_breed_with_img(img_path=img_path, model=model, feature_type = feature_type)
        predicted_breed =  breed.split('/')[-1].split('.')[-1]
        info = 'Dog detected with breed : '+ predicted_breed
        print('Dog detected with breed : ', predicted_breed)
    elif face_detector(img_path):
        breed = predict_dog_breed_with_img(img_path=img_path, model=model, feature_type = feature_type)
        predicted_breed =  breed.split('/')[-1].split('.')[-1]
        info = 'Human detected with most resembling dog breed: '+ predicted_breed
        print('Human detected with most resembling dog breed: ', predicted_breed)
    else:
        info = 'Image is neither Dog nor Human'
        print('Image is neither Dog nor Human')
        predicted_breed = 'Neither'
    return predicted_breed, info






