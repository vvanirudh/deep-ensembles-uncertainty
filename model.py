import tensorflow as tf
import numpy as np


class MLPGaussianRegressor():

    def __init__(self, args, sizes, model_scope):

        self.input_data = tf.placeholder(tf.float32, [None, sizes[0]])
        self.target_data = tf.placeholder(tf.float32, [None, sizes[0]])

        with tf.variable_scope(model_scope+'learning_rate'):
            self.lr = tf.Variable(args.learning_rate, trainable=False, name='learning_rate')

        self.weights = []
        self.biases = []
        # self.scales = []
        with tf.variable_scope(model_scope+'MLP'):
            for i in range(1, len(sizes)):
                self.weights.append(tf.Variable(tf.random_normal([sizes[i-1], sizes[i]], stddev=0.001), name='weights_'+str(i-1)))
                self.biases.append(tf.Variable(tf.random_normal([sizes[i]], stddev=0.001), name='biases_'+str(i-1)))
                # self.scales.append(tf.Variable(tf.random_normal([sizes[i]], stddev=0.001), name='scales_'+str(i-1)))

        x = self.input_data
        for i in range(0, len(sizes)-2):
            # z = tf.matmul(x, self.weights[i])
            # batch_mean, batch_var = tf.nn.moments(z, [0])
            # z = tf.nn.batch_normalization(z, batch_mean, batch_var, self.biases[i], self.scales[i], 1e-6)
            # x = tf.nn.relu(z)
            x = tf.nn.relu(tf.add(tf.matmul(x, self.weights[i]), self.biases[i]))

        self.output = tf.add(tf.matmul(x, self.weights[-1]), self.biases[-1])

        self.mean, self.raw_var = tf.split(1, 2, self.output)

        self.var = tf.log(1 + tf.exp(self.raw_var)) + 1e-6

        def gaussian_nll(mean_values, var_values, y):
            y_diff = tf.sub(y, mean_values)
            return 0.5*tf.reduce_mean(tf.log(var_values)) + 0.5*tf.reduce_mean(tf.div(tf.square(y_diff), var_values)) + 0.5*tf.log(2*np.pi)

        self.nll = gaussian_nll(self.mean, self.var, self.target_data)

        self.nll_gradients = tf.gradients(args.alpha * self.nll, self.input_data)[0]

        self.adversarial_input_data = tf.add(self.input_data, args.epsilon * tf.sign(self.nll_gradients))

        x_at = self.adversarial_input_data
        for i in range(0, len(sizes)-2):
            x_at = tf.nn.relu(tf.add(tf.matmul(x_at, self.weights[i]), self.biases[i]))

        output_at = tf.add(tf.matmul(x_at, self.weights[-1]), self.biases[-1])

        mean_at, raw_var_at = tf.split(1, 2, output_at)

        var_at = tf.log(1 + tf.exp(raw_var_at)) + 1e-6

        self.nll_at = gaussian_nll(mean_at, var_at, self.target_data)

        tvars = tf.trainable_variables()

        for v in tvars:
            print v.name
            print v.get_shape()

        self.gradients = tf.gradients(args.alpha * self.nll + (1 - args.alpha) * self.nll_at, tvars)

        self.clipped_gradients, _ = tf.clip_by_global_norm(self.gradients, args.grad_clip)

        optimizer = tf.train.RMSPropOptimizer(self.lr)

        self.train_op = optimizer.apply_gradients(zip(self.clipped_gradients, tvars))


class MLPDropoutGaussianRegressor():

    def __init__(self, args, sizes, model_scope):

        self.input_data = tf.placeholder(tf.float32, [None, sizes[0]])
        self.target_data = tf.placeholder(tf.float32, [None, sizes[0]])

        with tf.variable_scope(model_scope+'learning_rate'):
            self.lr = tf.Variable(args.learning_rate, trainable=False, name='learning_rate')

        self.dr = tf.placeholder(tf.float32)

        self.weights = []
        self.biases = []

        with tf.variable_scope(model_scope+'MLP'):
            for i in range(1, len(sizes)):
                self.weights.append(tf.Variable(tf.random_normal([sizes[i-1], sizes[i]], stddev=0.001), name='weights_'+str(i-1)))
                self.biases.append(tf.Variable(tf.random_normal([sizes[i]], stddev=0.001), name='biases_'+str(i-1)))

        x = self.input_data
        for i in range(0, len(sizes)-2):
            x = tf.nn.relu(tf.nn.xw_plus_b(x, self.weights[i], self.biases[i]))
            x = tf.nn.dropout(x, self.dr, noise_shape=[1, sizes[i+1]], name='dropout_layer'+str(i))

        # Not using adversarial data as we can't maintain the same model
        # due to the randomization in dropout.
        # Let's see how the model performs without adversarial data
        # I expect the probability distribution to be not smooth in x        

        self.output = tf.add(tf.matmul(x, self.weights[-1]), self.biases[-1])

        self.mean, self.raw_var = tf.split(1, 2, self.output)

        self.var = tf.log(1 + tf.exp(self.raw_var)) + 1e-6

        def gaussian_nll(mean_values, var_values, y):
            y_diff = tf.sub(y, mean_values)
            return 0.5*tf.reduce_mean(tf.log(var_values)) + 0.5*tf.reduce_mean(tf.div(tf.square(y_diff), var_values)) + 0.5*tf.log(2*np.pi)

        self.nll = gaussian_nll(self.mean, self.var, self.target_data)

        self.nll_gradients = tf.gradients(args.alpha * self.nll, self.input_data)[0]
        self.adversarial_input_data = tf.add(self.input_data, args.epsilon * tf.sign(self.nll_gradients))

        x_at = self.adversarial_input_data
        for i in range(0, len(sizes)-2):
            x_at = tf.nn.relu(tf.nn.xw_plus_b(x_at, self.weights[i], self.biases[i]))
            graph = tf.get_default_graph()
            mask = graph.get_tensor_by_name('dropout_layer'+str(i)+'/Floor:0')
            x_at = tf.mul(x_at, mask)

        output_at = tf.add(tf.matmul(x_at, self.weights[-1]), self.biases[-1])

        mean_at, raw_var_at = tf.split(1, 2, output_at)

        var_at = tf.log(1 + tf.exp(raw_var_at)) + 1e-6

        self.nll_at = gaussian_nll(mean_at, var_at, self.target_data)

        tvars = tf.trainable_variables()

        for v in tvars:
            print v.name
            print v.get_shape()

        self.gradients = tf.gradients(args.alpha * self.nll + (1 - args.alpha) * self.nll_at, tvars)

        self.clipped_gradients, _ = tf.clip_by_global_norm(self.gradients, args.grad_clip)

        optimizer = tf.train.RMSPropOptimizer(self.lr)

        self.train_op = optimizer.apply_gradients(zip(self.clipped_gradients, tvars))
