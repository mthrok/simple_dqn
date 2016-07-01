import logging
import datetime

import numpy as np
import tensorflow as tf

from tf_model import Model, Conv2D, Dense, ReLU, Flatten, L2, TrueDiv

logger = logging.getLogger(__name__)


class DeepQNetwork(object):
    def __init__(self, num_actions, args):
        self.num_actions = num_actions
        self.batch_size = args.batch_size
        self.discount_rate = args.discount_rate
        self.history_length = args.history_length
        self.screen_dim = (args.screen_height, args.screen_width)
        self.clip_error = args.clip_error
        self.min_reward = args.min_reward
        self.max_reward = args.max_reward
        self.batch_norm = args.batch_norm
        self.backend = args.backend
        self.device_id = args.device_id
        self.datatype = args.datatype
        self.learning_rate = args.learning_rate
        self.decay_rate = args.decay_rate
        self.target_steps = args.target_steps

        self._build_network()

    def _build_network(self):
        with tf.device(self._get_device_name()):
            with tf.variable_scope('model'):
                self.model = self._build_model()
            with tf.variable_scope('target'):
                self.target = self._build_model()
            with tf.name_scope('error'):
                self._build_error()
            with tf.name_scope('optimization'):
                self._build_optimization()
            with tf.name_scope('sync'):
                self._build_sync_ops()
            with tf.name_scope('summary'):
                self._build_summary_ops()

        self.session = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True))

        self.session.run(tf.initialize_all_variables())

        self._init_summary_writer()

        self._init_saver()

    def _build_model(self):
        if self.backend == 'gpu':
            input_shape = (None, self.history_length) + self.screen_dim
            data_format = 'NCHW'
        else:
            input_shape = (None, ) + self.screen_dim + (self.history_length, )
            data_format = 'NHWC'

        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        conv1 = Conv2D((8, 8, 32), 4, initializer, 'VALID', data_format)
        conv2 = Conv2D((4, 4, 64), 2, initializer, 'VALID', data_format)
        conv3 = Conv2D((3, 3, 64), 1, initializer, 'VALID', data_format)
        dense1 = Dense(512, initializer)
        dense2 = Dense(self.num_actions, initializer)

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

        input_tensor = tf.placeholder(dtype=self.datatype, shape=input_shape)

        model(input_tensor)
        return model

    def _get_device_name(self):
        return '/{}:{}'.format(self.backend, self.device_id)

    def _build_error(self):
        source = self.model.output_tensor
        target = tf.placeholder(dtype=self.datatype, shape=source.get_shape())
        self.l2 = L2(-self.clip_error, self.clip_error)
        self.l2(target, source)

    def _build_optimization(self):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        opt = tf.train.RMSPropOptimizer(
            learning_rate=self.learning_rate,
            decay=self.decay_rate)
        wrt = self.model.get_variables().values()
        self.optimization_op = opt.minimize(
            self.l2.error, global_step=self.global_step, var_list=wrt)
        self.optimizer = opt

    def _build_sync_ops(self):
        src_vars = self.model.get_variables().values()
        tgt_vars = self.target.get_variables().values()
        ops = [tgt.assign(src) for src, tgt in zip(src_vars, tgt_vars)]
        self.sync_op = tf.group(*ops, name='sync')

    def _build_summary_ops(self):
        self._build_net_summary_ops()
        self._build_training_summary_ops()

    def _build_training_summary_ops(self):
        self.summary_placeholder = tf.placeholder(dtype=self.datatype)
        self.summary_values = {
            'error': [],
            'reward': [],
            'steps': [],
        }
        self.training_summary_ops = {}
        for key in self.summary_values.keys():
            for attr in ['average', 'min', 'max']:
                name = '{}/{}'.format(key, attr)
                self.training_summary_ops[name] = tf.scalar_summary(
                    name, self.summary_placeholder)

    def _build_net_summary_ops(self):
        variables = self.model.get_variables().values()
        ops1 = [tf.histogram_summary('/'.join(var.name.split('/')[1:]), var)
                for var in variables]
        variables = self.model.get_output_tensors().values()
        ops2 = [tf.histogram_summary('/'.join(var.name.split('/')[1:]), var)
                for var in variables]
        self.net_summary_ops = ops1 + ops2

    def _init_summary_writer(self):
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        path = 'summary/{}'.format(now)
        self.writer = tf.train.SummaryWriter(path, self.session.graph)

    def _init_saver(self):
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)

    def train(self, minibatch, epoch):
        prestates, actions, rewards, poststates, terminals = minibatch
        assert len(prestates.shape) == 4
        assert len(poststates.shape) == 4
        assert len(actions.shape) == 1
        assert len(rewards.shape) == 1
        assert len(terminals.shape) == 1
        assert prestates.shape == poststates.shape
        assert prestates.shape[0] == actions.shape[0] == rewards.shape[0] == poststates.shape[0] == terminals.shape[0]

        # minibatches are in order of 'NCHW'
        # If using CPU, this must be transposed to 'NHWC'
        if self.backend == 'cpu':
            prestates = prestates.transpose((0, 2, 3, 1))
            poststates = poststates.transpose((0, 2, 3, 1))

        rewards = np.clip(rewards, self.min_reward, self.max_reward)

        step = self.train_iterations
        if self._is_sync_step(step):
            self._sync_models()
            self._summarize_net(step, prestates)
            self._summarize_training(step)

        maxpostq = self._get_max_post_q(poststates)

        preq = self._get_pre_q(prestates)

        # make copy of prestate Q-values as targets
        targets = preq.copy()

        # clip rewards between -1 and 1
        rewards = np.clip(rewards, self.min_reward, self.max_reward)

        # update Q-value targets for actions taken
        for i, action in enumerate(actions):
            if terminals[i]:
                targets[i, action] = float(rewards[i])
            else:
                targets[i, action] = float(rewards[i]) + self.discount_rate * maxpostq[i]

        # perform optimization
        cost = self._optimize_model(prestates, targets)
        self.summary_values['error'].append(cost)

        if self.callback:
            self.callback.on_train(cost)

    @property
    def train_iterations(self):
        return self.session.run(self.global_step)

    def _is_sync_step(self, train_iterations):
        return self.target_steps and train_iterations % self.target_steps == 0

    def _sync_models(self):
        self.session.run(self.sync_op)

    def _summarize_net(self, step, states):
        summaries = self.session.run(self.net_summary_ops, feed_dict={
            self.model.input_tensor: states})
        for summary in summaries:
            self.writer.add_summary(summary, step)

    def _summarize_training(self, step):
        for key, values in self.summary_values.items():
            if values:
                name = '{}/average'.format(key)
                self._summarize_scalar(step, name, value=np.mean(values))
                name = '{}/min'.format(key)
                self._summarize_scalar(step, name, value=np.min(values))
                name = '{}/max'.format(key)
                self._summarize_scalar(step, name, value=np.max(values))
                self.summary_values[key] = []

    def _summarize_scalar(self, step, name, value):
        summary = self.session.run(
            self.training_summary_ops[name],
            feed_dict={self.summary_placeholder: value})
        self.writer.add_summary(summary, step)

    def _get_max_post_q(self, poststates):
        postq = self.session.run(self.target.output_tensor, feed_dict={
            self.target.input_tensor: poststates})
        maxpostq = postq.max(axis=1)
        return maxpostq

    def _get_pre_q(self, prestates):
        preq = self.session.run(self.model.output_tensor, feed_dict={
            self.model.input_tensor: prestates})
        return preq

    def _optimize_model(self, prestates, targets):
        error = self.session.run(
            [self.l2.error, self.optimization_op], feed_dict={
                self.model.input_tensor: prestates,
                self.l2.target: targets
            })[0]
        return error

    def predict(self, states):
        # minibatch is full size, because Agent pass in such way
        assert states.shape == ((self.batch_size, self.history_length,) + self.screen_dim)

        # minibatches are in order of 'NCHW'
        # If using CPU, this must be transposed to 'NHWC'
        if self.backend == 'cpu':
            states = states.transpose((0, 2, 3, 1))

        # calculate Q-values for the states
        qvalues = self._get_pre_q(states)
        assert qvalues.shape == (self.batch_size, self.num_actions)

        return qvalues

    def load_weights(self, load_path):
        self.saver.restore(self.session, load_path)

    def save_weights(self, save_path):
        self.saver.save(self.session, save_path)
