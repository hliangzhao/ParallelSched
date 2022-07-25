"""
The policy network and the value network of DL2.
"""
import numpy as np
import tflearn
import tensorflow as tf
import parameter as param


class PolicyNetwork(object):
    def __init__(self, sess, scope, mode, logger):
        self.sess = sess
        self.state_dim = param.STATE_DIM
        self.action_dim = param.ACTION_DIM
        self.scope = scope
        self.mode = mode
        self.logger = logger

        self.inputs, self.outputs = self._create_nn()
        self.labels = tf.placeholder(tf.float32, [None, self.action_dim])
        self.action = tf.placeholder(tf.float32, [None, None])
        self.advantage = tf.placeholder(tf.float32, [None, 1])

        self.entropy = tf.reduce_mean(
            tf.multiply(self.outputs, tf.log(self.outputs + param.ENTROPY_EPS))
        )
        self.entropy_weight = param.ENTROPY_WEIGHT

        # set loss func
        if self.mode == "SL":
            if param.SL_LOSS_FUNCTION == param.LOSS_FUNCS[1]:
                self.loss = tf.reduce_mean(tflearn.mean_square(self.outputs, self.labels))
            elif param.SL_LOSS_FUNCTION == param.LOSS_FUNCS[0]:
                self.loss = tf.reduce_mean(tflearn.categorical_crossentropy(self.outputs, self.labels))
            elif param.SL_LOSS_FUNCTION == param.LOSS_FUNCS[2]:
                self.loss = tf.reduce_mean(tf.losses.absolute_difference(self.outputs, self.labels))
            else:
                raise ValueError("Not supported loss func")

        elif self.mode == "RL":
            self.loss = tf.reduce_mean(
                tf.multiply(
                    tf.log(tf.reduce_sum(tf.multiply(self.outputs, self.action), reduction_indices=1, keep_dims=True)),
                    -self.advantage
                )
            ) + self.entropy_weight * self.entropy

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        self.gradients = tf.gradients(self.loss, self.weights)

        # set optimizer
        self.lr = param.LEARNING_RATE
        if param.OPTIMIZER == "Adam":
            self.optimize = tf.train.AdamOptimizer(learning_rate=self.lr).apply_gradients(
                zip(self.gradients, self.weights)
            )
        elif param.OPTIMIZER == "RMSProp":
            self.optimize = tf.train.RMSPropOptimizer(learning_rate=self.lr).apply_gradients(
                zip(self.gradients, self.weights)
            )

        self.weights_phs = []
        for weight in self.weights:
            self.weights_phs.append(tf.placeholder(tf.float32, shape=weight.get_shape()))
        self.set_weights_op = []
        for idx, weights_ph in enumerate(self.weights_phs):
            self.set_weights_op.append(self.weights[idx].assign(weights_ph))

        self.loss_ring_buff = [0] * 20
        self.index_ring_buff = 0

    def _create_nn(self):
        with tf.variable_scope(self.scope):
            # type, arrival, progress, resource
            inputs = tflearn.input_data(shape=[None, self.state_dim[0], self.state_dim[1]], name="inputs")

            if param.JOB_CENTRAL_REPRESENTATION or param.ATTRIBUTE_CENTRAL_REPRESENTATION:
                if param.JOB_CENTRAL_REPRESENTATION:
                    fc_list_1 = []
                    for i in range(self.state_dim[1]):
                        if param.FIRST_LAYER_TANH:
                            fc1 = tflearn.fully_connected(inputs[:, :, i], self.state_dim[0], activation="tanh",
                                                          name="job_" + str(i))
                        else:
                            fc1 = tflearn.fully_connected(inputs[:, :, i], self.state_dim[0], activation="relu",
                                                          name="job_" + str(i))
                        if param.BATCH_NORMALIZATION:
                            fc1 = tflearn.batch_normalization(fc1, name="job_" + str(i) + "_bn")
                        fc_list_1.append(fc1)
                else:
                    j = 0
                    fc_list_1 = []
                    # INPUTS_GATE = [("TYPE",True), ("STAY",False), ("PROGRESS",False), ("DOM_RESR",False), ("WORKERS",True)]
                    for key, enable in param.INPUTS_GATE:
                        if enable:
                            if param.FIRST_LAYER_TANH:
                                fc1 = tflearn.fully_connected(inputs[:, j], param.SCHED_WINDOW_SIZE, activation="tanh",
                                                              name=key)
                            else:
                                fc1 = tflearn.fully_connected(inputs[:, j], param.SCHED_WINDOW_SIZE, activation="relu",
                                                              name=key)
                            if param.BATCH_NORMALIZATION:
                                fc1 = tflearn.batch_normalization(fc1, name=key + "_bn")
                            fc_list_1.append(fc1)
                            j += 1
                if len(fc_list_1) == 1:
                    merge_net_1 = fc_list_1[0]
                    if param.BATCH_NORMALIZATION:
                        merge_net_1 = tflearn.batch_normalization(merge_net_1)
                else:
                    merge_net_1 = tflearn.merge(fc_list_1, "concat", name="merge_net_1")
                    if param.BATCH_NORMALIZATION:
                        merge_net_1 = tflearn.batch_normalization(merge_net_1, name="merge_net_1_bn")
                dense_net_1 = tflearn.fully_connected(merge_net_1, param.NUM_NEURONS_PER_FCN, activation="relu",
                                                      name="dense_net_1")

            else:
                dense_net_1 = tflearn.fully_connected(inputs, param.NUM_NEURONS_PER_FCN, activation="relu",
                                                      name="dense_net_1")

            if param.BATCH_NORMALIZATION:
                dense_net_1 = tflearn.batch_normalization(dense_net_1, name="dense_net_1_bn")

            for i in range(1, param.NUM_FCN_LAYERS):
                dense_net_1 = tflearn.fully_connected(dense_net_1, param.NUM_NEURONS_PER_FCN, activation='relu',
                                                      name='dense_net_' + str(i + 1))
                if param.BATCH_NORMALIZATION:
                    dense_net_1 = tflearn.batch_normalization(dense_net_1, name='dense_net_' + str(i + 1) + 'bn')

            if param.JOB_CENTRAL_REPRESENTATION and param.NN_SHORTCUT_CONN:
                fc_list_2 = []
                for fc in fc_list_1:
                    merge_net_2 = tflearn.merge([fc, dense_net_1], 'concat')
                    if param.PS_WORKER:
                        if param.BUNDLE_ACTION:
                            fc2 = tflearn.fully_connected(merge_net_2, 3, activation="linear")
                        else:
                            fc2 = tflearn.fully_connected(merge_net_2, 2, activation="linear")
                    else:
                        fc2 = tflearn.fully_connected(merge_net_2, 1, activation="linear")
                    fc_list_2.append(fc2)

                if param.SKIP_TS:
                    fc2 = tflearn.fully_connected(dense_net_1, 1, activation="linear")
                    fc_list_2.append(fc2)

                merge_net_3 = tflearn.merge(fc_list_2, 'concat')
                outputs = tflearn.activation(merge_net_3, activation="softmax", name="policy_output")

            else:
                outputs = tflearn.fully_connected(dense_net_1, self.action_dim, activation="softmax",
                                                  name="policy_output")
            return input, outputs

    def get_sl_loss(self, inputs, labels):
        assert self.mode == "SL"
        return self.sess.run(
            [self.outputs, self.loss],
            feed_dict={self.inputs: inputs, self.labels: labels}
        )

    def predict(self, inputs):
        return self.sess.run(
            self.outputs, feed_dict={self.inputs: inputs}
        )

    def get_sl_gradients(self, inputs, labels):
        assert self.mode == "SL"
        return self.sess.run(
            [self.entropy, self.loss, self.gradients],
            feed_dict={self.inputs: inputs, self.labels: labels}
        )

    def get_rl_gradients(self, inputs, outputs, action, advantage):
        assert self.mode == "RL"
        return self.sess.run(
            [self.entropy, self.loss, self.gradients],
            feed_dict={self.inputs: inputs, self.outputs: outputs, self.action: action, self.advantage: advantage}
        )

    def apply_gradients(self, gradients):
        self.sess.run(
            self.optimize,
            feed_dict={r: p for r, p in zip(self.gradients, gradients)}
        )

    def set_weights(self, weights):
        self.sess.run(
            self.set_weights_op,
            feed_dict={r: p for r, p in zip(self.weights_phs, weights)}
        )

    def get_weights(self):
        return self.sess.run(self.weights)

    def get_num_weights(self):
        with tf.variable_scope(self.scope):
            total_param = 0
            for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope):
                shape = var.get_shape()
                var_param = 1
                for dim in shape:
                    var_param *= dim.value
                total_param += var_param
            return total_param

    def anneal_entropy_weight(self, step):
        if param.FIX_ENTROPY_WEIGHT:
            self.entropy_weight = param.ENTROPY_WEIGHT
        else:
            self.entropy_weight = max(param.MAX_ENTROPY_WEIGHT * 2 / (1 + np.exp(step / param.ANNEALING_TEMPERATURE)),
                                      0.1)


class ValueNetwork(object):
    def __init__(self, sess, scope, mode, logger):
        self.sess = sess
        self.state_dim = param.STATE_DIM
        self.action_dim = param.ACTION_DIM
        self.scope = scope
        self.mode = mode
        self.logger = logger

        self.inputs, self.outputs = self._create_nn()
        self.labels = tf.placeholder(tf.float32, [None, self.action_dim])
        self.action = tf.placeholder(tf.float32, [None, None])

        self.entropy_weight = param.ENTROPY_WEIGHT

        self.td_target = tf.placeholder(tf.float32, [None, 1])
        self.loss = tflearn.mean_square(self.outputs, self.td_target)

        self.weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
        self.gradients = tf.gradients(self.loss, self.weights)

        self.lr = param.LEARNING_RATE
        if param.OPTIMIZER == "Adam":
            self.optimize = tf.train.AdamOptimizer(learning_rate=self.lr).apply_gradients(
                zip(self.gradients, self.weights))
        elif param.OPTIMIZER == "RMSProp":
            self.optimize = tf.train.RMSPropOptimizer(learning_rate=self.lr).apply_gradients(
                zip(self.gradients, self.weights))

        self.weights_phs = []
        for weight in self.weights:
            self.weights_phs.append(tf.placeholder(tf.float32, shape=weight.get_shape()))
        self.set_weights_op = []
        for idx, weights_ph in enumerate(self.weights_phs):
            self.set_weights_op.append(self.weights[idx].assign(weights_ph))

    def _create_nn(self):
        with tf.variable_scope(self.scope):
            # type, arrival, progress, resource
            inputs = tflearn.input_data(shape=[None, self.state_dim[0], self.state_dim[1]],
                                        name="inputs")

            if param.JOB_CENTRAL_REPRESENTATION or param.ATTRIBUTE_CENTRAL_REPRESENTATION:
                if param.JOB_CENTRAL_REPRESENTATION:
                    fc_list_1 = []
                    for i in range(self.state_dim[1]):
                        if param.FIRST_LAYER_TANH:
                            fc1 = tflearn.fully_connected(inputs[:, :, i], self.state_dim[0], activation="tanh",
                                                          name="job_" + str(i))
                        else:
                            fc1 = tflearn.fully_connected(inputs[:, :, i], self.state_dim[0], activation="relu",
                                                          name="job_" + str(i))
                        if param.BATCH_NORMALIZATION:
                            fc1 = tflearn.batch_normalization(fc1, name="job_" + str(i) + "_bn")
                        fc_list_1.append(fc1)
                else:
                    j = 0
                    fc_list_1 = []
                    for (key,
                         enable) in param.INPUTS_GATE:  # INPUTS_GATE=[("TYPE",True), ("STAY",False), ("PROGRESS",False), ("DOM_RESR",False), ("WORKERS",True)]
                        if enable:
                            if param.FIRST_LAYER_TANH:
                                fc1 = tflearn.fully_connected(inputs[:, j], param.SCHED_WINDOW_SIZE, activation="tanh",
                                                              name=key)
                            else:
                                fc1 = tflearn.fully_connected(inputs[:, j], param.SCHED_WINDOW_SIZE, activation="relu",
                                                              name=key)
                            if param.BATCH_NORMALIZATION:
                                fc1 = tflearn.batch_normalization(fc1, name=key + "_bn")
                            fc_list_1.append(fc1)
                            j += 1
                if len(fc_list_1) == 1:
                    merge_net_1 = fc_list_1[0]
                    if param.BATCH_NORMALIZATION:
                        merge_net_1 = tflearn.batch_normalization(merge_net_1)
                else:
                    merge_net_1 = tflearn.merge(fc_list_1, 'concat', name="merge_net_1")
                    if param.BATCH_NORMALIZATION:
                        merge_net_1 = tflearn.batch_normalization(merge_net_1, name="merge_net_1_bn")
                dense_net_1 = tflearn.fully_connected(merge_net_1, param.NUM_NEURONS_PER_FCN, activation='relu',
                                                      name='dense_net_1')
            else:
                dense_net_1 = tflearn.fully_connected(inputs, param.NUM_NEURONS_PER_FCN, activation='relu',
                                                      name='dense_net_1')
            if param.BATCH_NORMALIZATION:
                dense_net_1 = tflearn.batch_normalization(dense_net_1, name='dense_net_1_bn')

            for i in range(1, param.NUM_FCN_LAYERS):
                dense_net_1 = tflearn.fully_connected(dense_net_1, param.NUM_NEURONS_PER_FCN, activation='relu',
                                                      name='dense_net_' + str(i + 1))
                if param.BATCH_NORMALIZATION:
                    dense_net_1 = tflearn.batch_normalization(dense_net_1, name='dense_net_' + str(i + 1) + 'bn')

            if param.JOB_CENTRAL_REPRESENTATION and param.NN_SHORTCUT_CONN:  # a more layer if critic adds shortcut
                fc2_list = []
                for fc in fc_list_1:
                    merge_net_2 = tflearn.merge([fc, dense_net_1], 'concat')
                    if param.PS_WORKER:
                        if param.BUNDLE_ACTION:
                            fc2 = tflearn.fully_connected(merge_net_2, 3, activation='relu')
                        else:
                            fc2 = tflearn.fully_connected(merge_net_2, 2, activation='relu')
                    else:
                        fc2 = tflearn.fully_connected(merge_net_2, 1, activation='relu')
                    fc2_list.append(fc2)

                if param.SKIP_TS:
                    fc2 = tflearn.fully_connected(dense_net_1, 1, activation='relu')
                    fc2_list.append(fc2)

                merge_net_3 = tflearn.merge(fc2_list, 'concat', name='merge_net_3')
                if param.BATCH_NORMALIZATION:
                    merge_net_3 = tflearn.batch_normalization(merge_net_3, name='merge_net_3_bn')
                output = tflearn.fully_connected(merge_net_3, 1, activation="linear", name="value_output")

            else:
                output = tflearn.fully_connected(dense_net_1, 1, activation="linear", name="value_output")

            return inputs, output

    def get_loss(self, inputs):
        return self.sess.run(self.loss, feed_dict={self.inputs: inputs})

    def predict(self, inputs):
        return self.sess.run(self.outputs, feed_dict={self.inputs: inputs})

    def get_rl_gradients(self, inputs, output, td_target):
        return self.sess.run(
            [self.loss, self.gradients],
            feed_dict={self.inputs: inputs, self.outputs: output, self.td_target: td_target}
        )

    def apply_gradients(self, gradients):
        self.sess.run(self.optimize, feed_dict={r: p for r, p in zip(self.gradients, gradients)})

    def set_weights(self, weights):
        self.sess.run(self.set_weights_op, feed_dict={r: p for r, p in zip(self.weights_phs, weights)})

    def get_weights(self):
        return self.sess.run(self.weights)

    def get_num_weights(self):
        with tf.variable_scope(self.scope):
            total_param = 0
            for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope):
                shape = var.get_shape()
                var_param = 1
                for dim in shape:
                    var_param *= dim.value
                total_param += var_param
            return total_param
