from extract_bottleneck_features import *
from keras.models import load_model
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

import cv2                
import matplotlib.pyplot as plt  

from extract_bottleneck_features import *

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

from keras.applications.resnet50 import ResNet50

# define ResNet50 model
ResNet50_model_ = ResNet50(weights='imagenet')

from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    
    
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

from keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model_.predict(img))

### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

def my_dog_detector(img_path = '', model = None, feature_type = None):
    
    
    if dog_detector(img_path):
        breed = predict_dog_breed_with_img(img_path=img_path, model=model, feature_type = feature_type)
        predicted_breed =  breed.split('/')[-1].split('.')[-1]
        print('Dog detected with breed : ', predicted_breed)
    elif face_detector(img_path):
        breed = predict_dog_breed_with_img(img_path=img_path, model=model, feature_type = feature_type)
        predicted_breed =  breed.split('/')[-1].split('.')[-1]
        print('Human detected with most resembling dog breed: ', predicted_breed)
    else:
        print('Image is neither Dog nor Human')
        predicted_breed = 'Neither'
    return predicted_breed

def load_trained_model():
    model_path = './saved_models/weights.best.Resnet50.hdf5'
    feature_type = 'Resnet50'
    model = load_model(model_path)
    
    return model, feature_type




