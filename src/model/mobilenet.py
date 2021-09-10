
import tensorflow as tf

def mobilenet_cifar10():
    return tf.keras.applications.mobilenet.MobileNet(
        input_shape=(32, 32, 3), alpha=0.5, depth_multiplier=1, dropout=0,
        include_top=True, weights=None, input_tensor=None, pooling=None,
        classes=10
    )
