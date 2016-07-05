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


def vanilla_dqn(num_actions, data_format):
    initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
    conv1 = Conv2D((8, 8, 32), 4, initializer, 'VALID', data_format)
    conv2 = Conv2D((4, 4, 64), 2, initializer, 'VALID', data_format)
    conv3 = Conv2D((3, 3, 64), 1, initializer, 'VALID', data_format)
    dense1 = Dense(512, initializer)
    dense2 = Dense(num_actions, initializer)

    model = Model()
    model.add_layer(TrueDiv(255.0), 'layer0/normalization')

    model.add_layer(conv1, 'layer1/conv2D')
    model.add_layer(ReLU(), 'layer1/ReLU')

    model.add_layer(conv2, 'layer2/conv2D')
    model.add_layer(ReLU(), 'layer2/ReLU')

    model.add_layer(conv3, 'layer3/conv2D')
    model.add_layer(ReLU(), 'layer3/ReLU')

    model.add_layer(Flatten(), 'layer4/flatten')

    model.add_layer(dense1, 'layer5/dense')
    model.add_layer(ReLU(), 'layer5/ReLU')

    model.add_layer(dense2, 'layer6/dense')
    return model


class QLearning(object):
    def __init__(self, model_factory, input_shape, discount_rate,
                 min_reward=None, max_reward=None, datatype='float32'):
        self.input_shape = input_shape
        self.discount_rate = discount_rate
        self.datatype = datatype
        self.min_reward = min_reward
        self.max_reward = max_reward

        self._build_network(model_factory)
        self._build_sync_ops()

    def _build_network(self, model_factory):
        with tf.variable_scope('pre_trans'):
            self.prestates = tf.placeholder(
                dtype=self.datatype, shape=self.input_shape, name='prestates')
            self.pre_trans_model = model_factory()
            self.pre_trans_model(self.prestates)
        with tf.variable_scope('post_trans'):
            self.poststates = tf.placeholder(
                dtype=self.datatype, shape=self.input_shape, name='poststates')
            self.post_trans_model = model_factory()
            self.post_trans_model(self.poststates)
        with tf.name_scope('q_value'):
            self._build_q_value()

    def _build_q_value(self):
        self.actions = tf.placeholder(
            dtype='int32', shape=(None,), name='actions')
        self.rewards = tf.placeholder(
            dtype=self.datatype, shape=(None,), name='rewards')
        self.terminals = tf.placeholder(
            dtype=self.datatype, shape=(None,), name='terminals')
        with tf.name_scope('future_rewrad'):
            post_q = tf.reduce_max(self.post_trans_model.output_tensor,
                                   reduction_indices=1, name='max_post_q')
            post_q = tf.mul(self.discount_rate, post_q,
                            name='discounted_max_post_q')
            post_q = tf.mul((1.0 - self.terminals), post_q,
                            name='masked_max_post_q')
            rewards = self.rewards
            if self.min_reward and self.max_reward:
                rewards = tf.clip_by_value(
                    rewards, self.min_reward,
                    self.max_reward, name='clipped_reward')
            future_reward = tf.add(rewards, post_q, name='future_reward')

        with tf.name_scope('target_q_value'):
            target = tf.identity(self.pre_trans_model.output_tensor)
            n_actions = target.get_shape().as_list()[1]

            future_reward = tf.reshape(future_reward, (-1, 1))
            future_reward = tf.tile(future_reward, tf.pack([1, n_actions]))

            mask_on = tf.one_hot(
                self.actions, depth=n_actions, on_value=1., off_value=0.)
            mask_off = tf.one_hot(
                self.actions, depth=n_actions, on_value=0., off_value=1.)

            self.target_q = tf.add(target * mask_off, future_reward * mask_on,
                                   name='target_q')

    def _build_sync_ops(self):
        src_vars = self.pre_trans_model.get_variables().values()
        tgt_vars = self.post_trans_model.get_variables().values()
        with tf.name_scope('sync'):
            ops = [tgt.assign(src) for src, tgt in zip(src_vars, tgt_vars)]
            self.sync_op = tf.group(*ops, name='sync')
