from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
import cv2

from detector import get_dog_names, face_detector, dog_detector
from preprocess import path_to_tensor, paths_to_tensor

from extract_bottleneck_features import *

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

def build_model(bottleneck)
