import numpy as np
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt

import ipdb

from model import MLPGaussianRegressor
from model import MLPDropoutGaussianRegressor

from utils import DataLoader_RegressionToy
from utils import DataLoader_RegressionToy_withKink
from utils import DataLoader_RegressionToy_sinusoidal


def main():

    parser = argparse.ArgumentParser()
    # Ensemble size
    parser.add_argument('--ensemble_size', type=int, default=10,
                        help='Size of the ensemble')
    # Maximum number of iterations
    parser.add_argument('--max_iter', type=int, default=5000,
                        help='Maximum number of iterations')
    # Batch size
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Size of batch')
    # Epsilon for adversarial input perturbation
    parser.add_argument('--epsilon', type=float, default=1e-2,
                        help='Epsilon for adversarial input perturbation')
    # Alpha for trade-off between likelihood score and adversarial score
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Trade off parameter for likelihood score and adversarial score')
    # Learning rate
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='Learning rate for the optimization')
    # Gradient clipping value
    parser.add_argument('--grad_clip', type=float, default=100.,
                        help='clip gradients at this value')
    # Learning rate decay
    parser.add_argument('--decay_rate', type=float, default=0.99,
                        help='Decay rate for learning rate')
    # Dropout rate (keep prob)
    parser.add_argument('--keep_prob', type=float, default=0.5,
                        help='Keep probability for dropout')
    args = parser.parse_args()
    train_ensemble(args)
    # train_dropout(args)


def ensemble_mean_var(ensemble, xs, sess):
    en_mean = 0
    en_var = 0

    for model in ensemble:
        feed = {model.input_data: xs}
        mean, var = sess.run([model.mean, model.var], feed)
        en_mean += mean
        en_var += var + mean**2

    en_mean /= len(ensemble)
    en_var /= len(ensemble)
    en_var -= en_mean**2
    return en_mean, en_var


def dropout_mean_var(model, xs, sess, args):
    en_mean = 0
    en_var = 0

    for i in range(args.ensemble_size):
        # NOTE using dropout at test time as well
        feed = {model.input_data: xs, model.dr: args.keep_prob}
        mean, var = sess.run([model.mean, model.var], feed)
        en_mean += mean
        en_var += var + mean**2

    en_mean /= args.ensemble_size
    en_var /= args.ensemble_size
    en_var -= en_mean**2
    return en_mean, en_var


def train_ensemble(args):
    # Layer sizes
    sizes = [1, 50, 50, 2]
    # Input data
    dataLoader = DataLoader_RegressionToy_sinusoidal(args)

    # ipdb.set_trace()

    ensemble = [MLPGaussianRegressor(args, sizes, 'model'+str(i)) for i in range(args.ensemble_size)]

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for model in ensemble:
            sess.run(tf.assign(model.output_mean, dataLoader.target_mean))
            sess.run(tf.assign(model.output_std, dataLoader.target_std))

        for itr in range(args.max_iter):
            # print itr
            for model in ensemble:
                # sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate**itr)))
                x, y = dataLoader.next_batch()

                feed = {model.input_data: x, model.target_data: y}
                _, nll, m, v = sess.run([model.train_op, model.nll, model.mean, model.var], feed)

                # ipdb.set_trace()

                if itr % 100 == 0:
                    sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** (itr/100))))
                    print 'itr', itr, 'nll', nll

        test_ensemble(ensemble, sess, dataLoader)


def test_ensemble(ensemble, sess, dataLoader):
    test_xs, test_ys = dataLoader.get_test_data()
    mean, var = ensemble_mean_var(ensemble, test_xs, sess)
    std = np.sqrt(var)
    upper = mean + 3*std
    lower = mean - 3*std

    test_xs_scaled = dataLoader.input_mean + dataLoader.input_std*test_xs

    plt.plot(test_xs_scaled, test_ys, 'b-')
    plt.plot(test_xs_scaled, mean, 'r-')
    plt.plot(test_xs_scaled, upper, 'g-')
    plt.plot(test_xs_scaled, lower, 'c-')
    plt.show()


def train_dropout(args):
    # Layer sizes
    sizes = [1, 50, 50, 2]
    # Input data
    dataLoader = DataLoader_RegressionToy_sinusoidal(args)

    # ipdb.set_trace()

    model = MLPDropoutGaussianRegressor(args, sizes, 'dropout_model')

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        sess.run(tf.assign(model.output_mean, dataLoader.target_mean))
        sess.run(tf.assign(model.output_std, dataLoader.target_std))
        for itr in range(args.max_iter):

            x, y = dataLoader.next_batch()

            feed = {model.input_data: x, model.target_data: y, model.dr: args.keep_prob}
            _, nll = sess.run([model.train_op, model.nll], feed)

            if itr % 100 == 0:
                sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** (itr/100))))
                print 'itr', itr, 'nll', nll

        test_dropout(model, sess, dataLoader, args)


def test_dropout(model, sess, dataLoader, args):
    test_xs, test_ys = dataLoader.get_test_data()
    mean, var = dropout_mean_var(model, test_xs, sess, args)
    std = np.sqrt(var)
    upper = mean + 3*std
    lower = mean - 3*std

    plt.plot(test_xs, test_ys, 'b-')
    plt.plot(test_xs, mean, 'r-')
    plt.plot(test_xs, upper, 'g-')
    plt.plot(test_xs, lower, 'c-')
    plt.show()

if __name__ == '__main__':
    main()
