
import tensorflow as tf

def mobilenetv2_cifar10():
    return tf.keras.applications.mobilenet_v2.MobileNetV2(
        input_shape=(32, 32, 3), alpha=0.65,
        include_top=True, weights=None, input_tensor=None, pooling=None,
        classes=10
    )
