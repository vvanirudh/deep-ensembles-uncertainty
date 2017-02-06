import tensorflow as tf


class MLPGaussianRegressor():

    def __init__(self, args, sizes, model_scope):

        self.input_data = tf.placeholder(tf.float32, [None, sizes[0]])
        self.target_data = tf.placeholder(tf.float32, [None, sizes[0]])

        with tf.variable_scope(model_scope+'learning_rate'):
            self.lr = tf.Variable(args.learning_rate, trainable=False, name='learning_rate')

        self.weights = []
        self.biases = []
        with tf.variable_scope(model_scope+'MLP'):
            for i in range(1, len(sizes)):
                self.weights.append(tf.Variable(tf.random_normal([sizes[i-1], sizes[i]], stddev=0.001)))
                self.biases.append(tf.Variable(tf.random_normal([sizes[i]], stddev=0.001)))

        x = self.input_data
        for i in range(0, len(sizes)-2):
            x = tf.nn.relu(tf.add(tf.matmul(x, self.weights[i]), self.biases[i]))

        output = tf.add(tf.matmul(x, self.weights[-1]), self.biases[-1])

        self.mean = tf.reshape(output[:, 0], [-1, 1])
        raw_var = tf.reshape(output[:, 1], [-1, 1])
        self.var = tf.log(1 + tf.exp(raw_var)) + 1e-6

        def get_lossfunc(mean_values, var_values, y):
            return 0.5*tf.log(var_values) + 0.5*tf.div(tf.square(tf.sub(y, mean_values)), var_values)

        lossfunc_vec = get_lossfunc(self.mean, self.var, self.target_data)
        self.nll = tf.reduce_mean(lossfunc_vec)

        self.nll_gradients = tf.gradients(args.alpha * lossfunc_vec, self.input_data)[0]

        self.adversarial_input_data = tf.add(self.input_data, args.epsilon * tf.sign(self.nll_gradients))

        x_at = self.adversarial_input_data
        for i in range(0, len(sizes)-2):
            x_at = tf.nn.relu(tf.add(tf.matmul(x_at, self.weights[i]), self.biases[i]))
        output_at = tf.add(tf.matmul(x_at, self.weights[-1]), self.biases[-1])
        mean_at = tf.reshape(output_at[:, 0], [-1, 1])
        raw_var_at = tf.reshape(output_at[:, 1], [-1, 1])
        var_at = tf.log(1 + tf.exp(raw_var_at)) + 1e-6

        lossfunc_vec_at = get_lossfunc(mean_at, var_at, self.target_data)
        self.nll_at = tf.reduce_mean(lossfunc_vec_at)

        tvars = tf.trainable_variables()
        self.gradients = tf.gradients(args.alpha * self.nll + (1 - args.alpha) * self.nll_at, tvars)

        optimizer = tf.train.AdamOptimizer(self.lr)

        self.train_op = optimizer.apply_gradients(zip(self.gradients, tvars))
