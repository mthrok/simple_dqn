import logging
import datetime

import numpy as np

import luchador
from luchador.nn import (
    DeepQLearning,
    Input,
    SSE2,
    NeonRMSProp as Optimizer,
    Session,
    SummaryWriter,
    scope,
)
from luchador.nn.util import get_model


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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

        self.train_iterations = 0
        self.data_format = luchador.get_nn_conv_format()

        self._build_network()

    def _build_network(self):
        if self.data_format == 'NHWC':
            input_shape = (self.batch_size,) + self.screen_dim + (self.history_length,)
        else:
            input_shape = (self.batch_size, self.history_length) + self.screen_dim

        def model_maker():
            dqn = (
                get_model('image_normalizer', denom=255.0) +
                get_model('vanilla_dqn', n_actions=self.num_actions)
            )
            dqn(Input(shape=input_shape))
            return dqn

        self.ql = DeepQLearning(
            self.discount_rate, self.min_reward, self.max_reward)
        self.ql.build(model_maker)

        sse2 = SSE2(min_delta=-self.clip_error, max_delta=self.clip_error)
        self.sse_error = sse2(self.ql.target_q, self.ql.pre_trans_net.output)

        rmsprop = Optimizer(self.learning_rate, decay=self.decay_rate)
        params = self.ql.pre_trans_net.get_parameter_variables()
        with scope.name_scope('optimization'):
            self.update_op = rmsprop.minimize(self.sse_error, wrt=params.values())

        self.session = Session()
        self.session.initialize()

        now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        writer = SummaryWriter('./snapshots/{}/{}'.format(luchador.get_nn_backend(), now))
        writer.add_graph(self.session.graph, 0)

        stats = ['min', 'average', 'max']
        metrics = ['error', 'steps', 'reward']
        outputs = self.ql.pre_trans_net.get_output_tensors()
        writer.register('pre_trans_network_params', 'histogram', params.keys())
        writer.register('pre_trans_network_outputs', 'histogram', outputs.keys())
        writer.register('training', 'histogram',
                        ['training/{}'.format(metric) for metric in metrics])
        writer.register('training_steps', 'scalar',
                        ['steps/{}'.format(stat) for stat in stats])
        writer.register('training_error', 'scalar',
                        ['error/{}'.format(stat) for stat in stats])
        writer.register('training_reward', 'scalar',
                        ['reward/{}'.format(stat) for stat in stats])
        self.writer = writer
        self.summary_values = {
            'error': [],
            'reward': [],
            'steps': [],
        }

    '''
    def _get_device_name(self):
        return '/{}:{}'.format(self.backend, self.device_id)

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
        ops = {}
        for key in self.summary_values.keys():
            name = 'training/{}'.format(key)
            ops[name] = tf.histogram_summary(name, self.summary_placeholder)
            for attr in ['average', 'min', 'max']:
                name = '{}/{}'.format(key, attr)
                ops[name] = tf.scalar_summary(name, self.summary_placeholder)
        self.training_summary_ops = ops

    def _init_summary_writer(self):
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        path = 'snapshots/{}'.format(now)
        self.writer = tf.train.SummaryWriter(path, self.session.graph)

    def _init_saver(self):
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)
    '''
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
        # If using CPU in tensorflow, this must be transposed to 'NHWC'
        if self.data_format == 'NHWC':
            prestates = prestates.transpose((0, 2, 3, 1))
            poststates = poststates.transpose((0, 2, 3, 1))

        step = self.train_iterations
        self.train_iterations += 1
        if self._is_sync_step(step):
            self._sync_models()
            self._summarize_net(step, prestates)
            self._summarize_training(step)

        # perform optimization
        cost = self._optimize_model(prestates, actions, rewards, poststates, terminals)
        self.summary_values['error'].append(cost)

        if self.callback:
            self.callback.on_train(cost)

    def _is_sync_step(self, train_iterations):
        return self.target_steps and train_iterations % self.target_steps == 0

    def _sync_models(self):
        self.session.run(name='sync', updates=self.ql.sync_op)

    def _summarize_net(self, step, states):
        params = self.ql.pre_trans_net.get_parameter_variables()
        outputs = self.ql.pre_trans_net.get_output_tensors()
        params_vals = self.session.run(
            name='params', outputs=params.values())
        output_vals = self.session.run(
            name='outputs', outputs=outputs.values(),
            inputs={self.ql.pre_states: states})
        self.writer.summarize('pre_trans_network_params', step, params_vals)
        self.writer.summarize('pre_trans_network_outputs', step, output_vals)

    def _summarize_training(self, step):
        metrics = ['error', 'steps', 'reward']
        values = [self.summary_values[metric] for metric in metrics]
        self.writer.summarize('training', step, values)

        for metric in metrics:
            summary_values = self.summary_values[metric]
            if not summary_values:
                continue
            values = [
                np.min(summary_values),
                np.mean(summary_values),
                np.max(summary_values),
            ]
            self.writer.summarize('training_{}'.format(metric), step, values)
            self.summary_values[metric] = []

    '''
    def _summarize(self, step, name, value):
        summary = self.session.run(
            self.training_summary_ops[name],
            feed_dict={self.summary_placeholder: value})
        self.writer.add_summary(summary, step)
    '''
    def _get_pre_q(self, prestates):
        preq = self.session.run(
            name='predict',
            outputs=[self.ql.predicted_q],
            inputs={self.ql.pre_states: prestates}
        )[0]
        return preq

    def _optimize_model(self, prestates, actions, rewards, poststates, terminals):
        rewards = rewards.astype(np.float32)
        error = self.session.run(
            name='optimization',
            outputs=self.sse_error,
            inputs={
                self.ql.pre_states: prestates,
                self.ql.actions: actions,
                self.ql.rewards: rewards,
                self.ql.post_states: poststates,
                self.ql.terminals: terminals,
            },
            updates=self.update_op)
        return error

    def predict(self, states):
        # minibatch is full size, because Agent pass in such way
        assert states.shape == (self.batch_size, self.history_length,) + self.screen_dim
        if self.data_format == 'NHWC':
            states = states.transpose((0, 2, 3, 1))

        # calculate Q-values for the states
        qvalues = self._get_pre_q(states)
        assert qvalues.shape == (self.batch_size, self.num_actions)

        return qvalues

    def load_weights(self, load_path):
        # self.saver.restore(self.session, load_path)
        pass

    def save_weights(self, save_path):
        # self.saver.save(self.session, save_path)
        pass
