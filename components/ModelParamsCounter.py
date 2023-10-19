import tensorflow as tf

class ModelParamsCounter:
    @staticmethod
    def get_total_params(model):
        return model.count_params()  
    
    @staticmethod
    def get_model(model_name):
        if model_name == "resnet50":
            return tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
        elif model_name == "resnet101":
            return tf.keras.applications.ResNet101(weights='imagenet', include_top=False)
        elif model_name == "resnet152":
            return tf.keras.applications.ResNet152(weights='imagenet', include_top=False)
        elif model_name == "densenet201":
            return tf.keras.applications.DenseNet201(weights='imagenet', include_top=False)
        elif model_name == "vgg16":
            return tf.keras.applications.VGG16(weights='imagenet', include_top=False)
        elif model_name == "mobilenet":
            return tf.keras.applications.MobileNet(weights='imagenet', include_top=False)
        elif model_name == "EfficientNetB0":
            return tf.keras.applications.EfficientNetB0(weights='imagenet', include_top=False)
        elif model_name == "mobilenetv2":
            return tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)
        else:
            raise ValueError(f"Invalid model name {model_name}")
