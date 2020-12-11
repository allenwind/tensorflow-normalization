import tensorflow as tf
from tensorflow.keras.layers import *

class LayerNormalization(tf.keras.layers.Layer):
    """在特征维度的归一化"""

    def __init__(
        self,
        epsilon=1e-3,
        center=True,
        scale=True,
        trainable=True,
        **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon # 避免出现0方差
        self.center = center
        self.scale = scale
        self.trainable = trainable

    def build(self, input_shape):
        shape = (input_shape[-1],)
        # 使用简单的初始化
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                initializer="zeros",
                trainable=self.trainable,
                name="beta"
            )
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                initializer="ones",
                trainable=self.trainable,
                name="gamma"
            )

    def call(self, inputs):
        x = inputs
        # norm(x) * gamma + beta
        if self.center:
            mean = tf.reduce_mean(x, axis=-1, keepdims=True)
            x = x - mean
        if self.scale:
            variance = tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
            std = tf.sqrt(variance + self.epsilon)
            x = x / std * self.gamma
        if self.center:
            x = x + self.beta
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

class BatchSequenceNormalization(tf.keras.layers.Layer):
    """在一个batch上序列方向即axis=1计算均值和方差然后再标准化，
    用在时间序列相关问题上。"""

    def __init__(
        self,
        epsilon=1e-3,
        center=True,
        scale=True,
        trainable=True,
        **kwargs):
        super(BatchSequenceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon # 避免出现0方差
        self.center = center
        self.scale = scale
        self.trainable = trainable

    def build(self, input_shape):
        shape = (input_shape[-1],)
        # 使用简单的初始化
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                initializer="zeros",
                trainable=self.trainable,
                name="beta"
            )
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                initializer="ones",
                trainable=self.trainable,
                name="gamma"
            )

    def call(self, inputs, mask=None):
        if mask is None:
            mask = 1.0
        else:
            mask = tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)
        x = inputs
        if self.center:
            # (1, 1, hdims)
            mean = tf.reduce_sum(inputs, axis=[0, 1], keepdims=True) / tf.reduce_sum(mask)
            x = x - mean
        if self.scale:
            variance = tf.reduce_sum(tf.square(x), axis=[0, 1], keepdims=True) / tf.reduce_sum(mask)
            std = tf.sqrt(variance + self.epsilon)
            x = x / std * self.gamma
        if self.center:
            x = x + self.beta
        return x

    def compute_output_shape(self, input_shape):
        return input_shape
