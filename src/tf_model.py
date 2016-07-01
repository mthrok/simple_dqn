import logging
from collections import OrderedDict

import tensorflow as tf

logger = logging.getLogger(__name__)


class LayerConfig(object):
    def __init__(self, layer, scope):
        self.layer = layer
        self.scope = scope
        self.input_tensor = None
        self.output_tensor = None


class Model(object):
    def __init__(self):
        self.layer_configs = []

    def add_layer(self, layer, scope):
        self.layer_configs.append(LayerConfig(layer, scope))

    def __call__(self, input_tensor):
        self.input_tensor = input_tensor
        tensor = input_tensor
        for cfg in self.layer_configs:
            cfg.input_tensor = tensor
            with tf.variable_scope(cfg.scope):
                tensor = cfg.layer(tensor)
            cfg.output_tensor = tensor
        self.output_tensor = tensor
        return tensor

    def get_variables(self):
        ret = OrderedDict()
        for cfg in self.layer_configs:
            for name, var in cfg.layer.get_variables().items():
                name = '{}/{}'.format(cfg.scope, name)
                ret[name] = var
        return ret

    def get_output_tensors(self):
        ret = OrderedDict()
        for cfg in self.layer_configs:
            name = '{}/output'.format(cfg.scope)
            ret[name] = cfg.output_tensor
        return ret


class BaseLayer(object):
    def get_variables(self):
        return {}

    def __call__(self, input_tensor):
        raise NotImplementedError(
            '`__call__` is not implemented for {}'.format(type(self)))


class Conv2D(BaseLayer):
    def __init__(self, shape, strides, initializer,
                 padding='VALID', data_format='NCHW'):
        assert len(shape) == 3, '`fshape` must be (h, w, out_ch)'
        assert isinstance(strides, int), '`strides` must be int'

        self.shape = shape
        self.strides = strides
        self.padding = padding
        self.initializer = initializer
        self.format = data_format

    def __call__(self, input_tensor):
        in_shape = input_tensor.get_shape().as_list()
        in_ch = in_shape[1] if self.format == 'NCHW' else in_shape[3]
        self._instantiate_variables(in_ch)

        strides = ((1, 1, self.strides, self.strides)
                   if self.format == 'NCHW' else
                   (1, self.strides, self.strides, 1))
        out = tf.nn.conv2d(
            input_tensor, self.weight, strides=strides,
            padding=self.padding, use_cudnn_on_gpu=True,
            data_format=self.format, name='conv')
        out = tf.nn.bias_add(
            out, self.bias, data_format=self.format, name='output')
        return out

    def _instantiate_variables(self, in_ch):
        b_shape = (self.shape[2], )
        w_shape = (self.shape[0], self.shape[1], in_ch, self.shape[2])
        self.bias = tf.get_variable(
            name='bias', shape=b_shape, initializer=self.initializer)
        self.weight = tf.get_variable(
            name='weight', shape=w_shape, initializer=self.initializer)

    def get_variables(self):
        return OrderedDict((('weight', self.weight), ('bias', self.bias)))


class Dense(BaseLayer):
    def __init__(self, n_out, initializer):
        self.n_out = n_out
        self.initializer = initializer

    def __call__(self, input_tensor):
        in_shape = input_tensor.get_shape().as_list()
        self.n_in = in_shape[1]
        self._instantiate_variables()

        out = tf.matmul(input_tensor, self.weight, name='prod')
        out = tf.add(out, self.bias, name='output')
        return out

    def _instantiate_variables(self):
        b_shape, w_shape = (self.n_out,), (self.n_in, self.n_out,)
        self.bias = tf.get_variable(
            name='bias', shape=b_shape, initializer=self.initializer)
        self.weight = tf.get_variable(
            name='weight', shape=w_shape, initializer=self.initializer)

    def get_variables(self):
        return OrderedDict((('weight', self.weight), ('bias', self.bias)))


class ReLU(BaseLayer):
    def __init__(self):
        pass

    def __call__(self, input_tensor):
        return tf.nn.relu(input_tensor, name='ouptut')


class Flatten(BaseLayer):
    def __init__(self):
        pass

    def __call__(self, input_tensor):
        in_shape = input_tensor.get_shape().as_list()
        n_out = reduce(lambda prod, dim: prod*dim, in_shape[1:], 1)
        return tf.reshape(input_tensor, (-1, n_out), 'output')


class TrueDiv(BaseLayer):
    def __init__(self, denom):
        self.denom = denom

    def __call__(self, input_tensor):
        return tf.truediv(input_tensor, self.denom, 'output')


class L2(object):
    def __init__(self, min_delta=None, max_delta=None):
        self.max_delta = max_delta
        self.min_delta = min_delta

    def __call__(self, target, source):
        self.target, self.source = target, source
        delta = tf.sub(target, source, 'delta')
        if self.max_delta and self.min_delta:
            delta = tf.clip_by_value(
                delta, self.min_delta, self.max_delta, name='clipped_delta')
        error = tf.square(delta, name='squared_error')
        error = tf.truediv(tf.reduce_sum(error, reduction_indices=1), 2.0,
                           name='L2_error')
        self.error = tf.reduce_mean(error, name='mean_L2_error')
        return self.error
