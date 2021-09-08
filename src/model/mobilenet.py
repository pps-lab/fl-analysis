
import tensorflow as tf

def mobilenet_cifar10():
    return tf.keras.applications.mobilenet.MobileNet(
        input_shape=(32, 32, 3), alpha=1.0, depth_multiplier=1,
        include_top=True, weights=None, input_tensor=None, pooling=None,
        classes=10, classifier_activation='softmax'
    )
