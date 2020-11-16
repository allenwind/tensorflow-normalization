import tensorflow as tf

class BatchNormalization(tf.keras.layers.Layer):
    pass


class LayerNormalization(tf.keras.layers.Layer):

    def __init__(self,
                 hidden_activation='linear',
                 hidden_initializer='glorot_uniform',
                 **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.hidden_activation = tf.keras.activations.get(hidden_activation)
        self.hidden_initializer = tf.keras.initializers.get(hidden_initializer)
        self.epsilon = tf.keras.backend.epsilon() ** 2

    def build(self, input_shape):
        dims = input_shape[-1]
        self.gamma = self.add_weight(shape=(dims,),
                                     initializer='ones',
                                     name='gamma')
        self.beta = self.add_weight(shape=(dims,),
                                    initializer='zeros',
                                    name='beta')

    def call(self, inputs):
        """如果是条件Layer Norm，则默认以list为输入，第二个是condition
        """
        beta, gamma = self.beta, self.gamma
        mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=-1, keepdims=True)
        std = tf.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        outputs = outputs * gamma + beta
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

class GroupNormalization(Layer):
    
    def __init__(self, group_axis=(-1,)):
        self.group_axis = group_axis
        self.epsilon = tf.keras.backend.epsilon() ** 2

    def build(self, input_shape):
        pass

class InstanceNormalization(Layer):
    pass

class BatchSequenceNormalization(Layer):
    # TODO mask
    # TODO reduce_variance

    def __init__(self, **kwargs):
        super(BatchSequenceNormalization, self).__init__(**kwargs)
        self.epsilon = tf.keras.backend.epsilon() ** 2

    def build(self, input_shape):
        dims = input_shape[-1]
        self.gamma = self.add_weight(shape=(dims,), initializer="ones", name="beta")
        self.beta = self.add_weight(shape=(dims,), initializer="zeros", name="gamma")

    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=[0, -1], keepdims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=[0, -1], keepdims=True)
        std = tf.sqrt(variance, self.epsilon)
        outputs = (inputs - mean) / std
        outputs = outputs * self.gamma + self.beta
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

class ConditionalLayerNormalization(Layer):
    """(Conditional) Layer Normalization
    hidden_*系列参数仅为有条件输入时(conditional=True)使用
    """
    def __init__(self,
                 conditional=False,
                 hidden_units=None,
                 hidden_activation='linear',
                 hidden_initializer='glorot_uniform',
                 **kwargs):
        super(LayerNormalization, self).__init__(**kwargs)
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_activation = activations.get(hidden_activation)
        self.hidden_initializer = initializers.get(hidden_initializer)
        self.epsilon = K.epsilon() * K.epsilon()

    def build(self, input_shape):
        super(LayerNormalization, self).build(input_shape)

        if self.conditional:
            shape = (input_shape[0][-1], )
        else:
            shape = (input_shape[-1], )

        self.gamma = self.add_weight(shape=shape,
                                     initializer='ones',
                                     name='gamma')
        self.beta = self.add_weight(shape=shape,
                                    initializer='zeros',
                                    name='beta')

        if self.conditional:

            if self.hidden_units is not None:
                self.hidden_dense = Dense(
                    units=self.hidden_units,
                    activation=self.hidden_activation,
                    use_bias=False,
                    kernel_initializer=self.hidden_initializer)

            self.beta_dense = Dense(units=shape[0],
                                    use_bias=False,
                                    kernel_initializer='zeros')
            self.gamma_dense = Dense(units=shape[0],
                                     use_bias=False,
                                     kernel_initializer='zeros')

    def call(self, inputs):
        """如果是条件Layer Norm，则默认以list为输入，第二个是condition
        """
        if self.conditional:
            inputs, cond = inputs
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)
            for _ in range(K.ndim(inputs) - K.ndim(cond)):
                cond = K.expand_dims(cond, 1)
            beta = self.beta_dense(cond)
            gamma = self.gamma_dense(cond)
            beta, gamma = self.beta + beta, self.gamma + gamma
        else:
            beta, gamma = self.beta, self.gamma

        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        outputs = outputs * gamma + beta
        return outputs

    def get_config(self):
        config = {
            'conditional': self.conditional,
            'hidden_units': self.hidden_units,
            'hidden_activation': activations.serialize(self.hidden_activation),
            'hidden_initializer': initializers.serialize(self.hidden_initializer),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
