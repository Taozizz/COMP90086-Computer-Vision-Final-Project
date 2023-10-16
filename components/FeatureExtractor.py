import os
import numpy as np
import pickle
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from keras.preprocessing import image
from tensorflow.keras.applications import (
    ResNet50,
    ResNet101,
    ResNet152,
    DenseNet201,
    VGG16,
    MobileNet,
)

class FeatureExtractor:
    def __init__(self, model_name, base_output_dir):
        self.model_name = model_name
        self.model, self.preprocess_func = self.get_model_and_preprocess(model_name)
        self.base_output_dir = base_output_dir
    
    @staticmethod
    def get_model_and_preprocess(model_name):
        if model_name == "resnet50":
            model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
            preprocess_func = tf.keras.applications.resnet50.preprocess_input
        elif model_name == "resnet101":
            model = ResNet101(weights='imagenet', include_top=False, pooling='avg')
            preprocess_func = tf.keras.applications.resnet.preprocess_input
        elif model_name == "resnet152":
            model = ResNet152(weights='imagenet', include_top=False, pooling='avg')
            preprocess_func = tf.keras.applications.resnet.preprocess_input
        elif model_name == "densenet201":
            model = DenseNet201(weights='imagenet', include_top=False, pooling='avg')
            preprocess_func = tf.keras.applications.densenet.preprocess_input
        elif model_name == "vgg16":
            model = VGG16(weights='imagenet', include_top=False, pooling='avg')
            preprocess_func = tf.keras.applications.vgg16.preprocess_input
        elif model_name == "mobilenet":
            model = MobileNet(weights='imagenet', include_top=False, pooling='avg')
            preprocess_func = tf.keras.applications.mobilenet.preprocess_input
        else:
            valid_model_names = ["resnet50", "resnet101", "resnet152", "densenet201", "vgg16", "mobilenet"]
            raise ValueError(f"Invalid model name. Valid model names are: {', '.join(valid_model_names)}")

        
        return model, preprocess_func

    def extract_and_save_features(self, data_dict, set_name):
        # Modify the model_dir to include the model name as a subdirectory
        model_dir = os.path.join(self.base_output_dir, self.model_name, set_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # New dictionary to hold features
        features_dict = {}
        
        for key, value in data_dict.items():
            if isinstance(value, list):  # Handling case where value is a list of image paths
                features_dict[key] = [self.extract_features_from_image(img_path) for img_path in value]
            else:  # Handling case where value is a single image path
                features_dict[key] = self.extract_features_from_image(value)
        
        # Save the features file directly under the model and set name directories
        output_path = os.path.join(model_dir, f"{set_name}_features.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(features_dict, f)
    
    def extract_features_from_image(self, img_path, target_size=(224, 224)):
        # Extract features from a single image
        img = img_to_array(load_img(img_path, target_size=target_size))
        img = np.expand_dims(img, axis=0)  # Adding batch dimension
        img_preprocessed = self.preprocess_func(img)
        features = self.model.predict(img_preprocessed)
        return features