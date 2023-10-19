import os
import numpy as np
import pickle
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from keras.preprocessing import image
from tensorflow.keras.applications import (
    ResNet50,
    ResNet101,
    ResNet152,
    DenseNet201,
    VGG16,
    MobileNet,
    MobileNetV2,
    EfficientNetB0,
    NASNetMobile,
)

class FeatureExtractor:
    def __init__(self, model_name, base_output_dir, fine_tune=False):
        self.model_name = model_name
        self.model, self.preprocess_func = self.get_model_and_preprocess(model_name)
        self.base_output_dir = base_output_dir
    
    @staticmethod
    def get_model_and_preprocess(model_name, fine_tune=False):
        if model_name == "resnet50":
            model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
            preprocess_func = tf.keras.applications.resnet50.preprocess_input
            
            if fine_tune:
                for layer in model.layers:
                    layer.trainable = True
        
        elif model_name == "EfficientNetB0":
            model = EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')
            preprocess_func = tf.keras.applications.efficientnet.preprocess_input
        elif model_name == "NASNetMobile":
            model = NASNetMobile(weights='imagenet', include_top=False, pooling='avg')
            preprocess_func = tf.keras.applications.nasnet.preprocess_input
        elif model_name == "mobilenetv2": 
            model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
            preprocess_func = tf.keras.applications.resnet.preprocess_input           
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
            valid_model_names = ["resnet18", "resnet50", "resnet101", "resnet152", "densenet201", "vgg16", "mobilenet"]
            raise ValueError(f"Invalid model name. Valid model names are: {', '.join(valid_model_names)}")

        
        return model, preprocess_func

    def extract_and_save_features(self, data_dict, set_name):
        # Modify the model_dir to include the model name as a subdirectory
        model_dir = os.path.join(self.base_output_dir, self.model_name, set_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # New dictionaries to hold features
        left_features_dict = {}
        right_features_dict = {}
        
        for left_img_path, right_img_path in data_dict.items():
            # Extract features for the left and right images
            left_features_dict[left_img_path] = self.extract_features_from_image(left_img_path)
            right_features_dict[right_img_path] = self.extract_features_from_image(right_img_path)
        
        # Save the features files directly under the model and set name directories
        left_output_path = os.path.join(model_dir, f"{set_name}_left_features.pkl")
        right_output_path = os.path.join(model_dir, f"{set_name}_right_features.pkl")
        with open(left_output_path, 'wb') as f:
            pickle.dump(left_features_dict, f)
        with open(right_output_path, 'wb') as f:
            pickle.dump(right_features_dict, f)

    
    def extract_features_from_image(self, img_path, target_size=(224, 224)):
        # Extract features from a single image
        img = img_to_array(load_img(img_path, target_size=target_size))
        img = np.expand_dims(img, axis=0)  # Adding batch dimension
        img_preprocessed = self.preprocess_func(img)
        features = self.model.predict(img_preprocessed)
        return features
    
    def extract_and_save_features_for_test(self, data_dict, set_name):
        model_dir = os.path.join(self.base_output_dir, self.model_name, set_name)
        os.makedirs(model_dir, exist_ok=True)

        left_features_dict = {}
        right_features_dict = {}

        for left_img_path, right_img_series in data_dict.items():
            # Extract features for the left image
            left_features_dict[left_img_path] = self.extract_features_from_image(left_img_path)

            for right_img_path in right_img_series:
                # Check if the feature has already been extracted to avoid duplicate work
                if right_img_path not in right_features_dict:
                    right_features_dict[right_img_path] = self.extract_features_from_image(right_img_path)

        # Save the features
        left_output_path = os.path.join(model_dir, f"{set_name}_left_features.pkl")
        right_output_path = os.path.join(model_dir, f"{set_name}_right_features.pkl")

        with open(left_output_path, 'wb') as f:
            pickle.dump(left_features_dict, f)

        with open(right_output_path, 'wb') as f:
            pickle.dump(right_features_dict, f)
            
    def fine_tune(self, train_data, train_labels, epochs, batch_size, validation_data=None):
        # Compile the model for fine-tuning
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5) # Using a smaller learning rate for fine-tuning
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(train_data, train_labels, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

