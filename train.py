import numpy as np
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt

import ipdb

from model import MLPGaussianRegressor
from utils import DataLoader_RegressionToy
from utils import DataLoader_RegressionToy_withKink


def main():

    parser = argparse.ArgumentParser()
    # Ensemble size
    parser.add_argument('--ensemble_size', type=int, default=10,
                        help='Size of the ensemble')
    # Maximum number of iterations
    parser.add_argument('--max_iter', type=int, default=5000,
                        help='Maximum number of iterations')
    # Batch size
    parser.add_argument('--batch_size', type=int, default=10,
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
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    # Learning rate decay
    parser.add_argument('--decay_rate', type=float, default=0.99,
                        help='Decay rate for learning rate')
    args = parser.parse_args()
    train(args)


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


def train(args):
    # Layer sizes
    sizes = [1, 20, 20, 2]
    # Input data
    dataLoader = DataLoader_RegressionToy_withKink(args)

    # ipdb.set_trace()

    ensemble = [MLPGaussianRegressor(args, sizes, 'model'+str(i)) for i in range(args.ensemble_size)]

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for itr in range(args.max_iter):
            # print itr
            for model in ensemble:
                # sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate**itr)))
                x, y = dataLoader.next_batch()

                feed = {model.input_data: x, model.target_data: y}
                _, nll = sess.run([model.train_op, model.nll], feed)

                if itr % 100 == 0:
                    print 'itr', itr, 'nll', nll

        test(ensemble, sess, args)


def test(ensemble, sess, args):
    testDataLoader = DataLoader_RegressionToy_withKink(args, infer=False)
    test_xs, test_ys = testDataLoader.get_data()
    mean, var = ensemble_mean_var(ensemble, test_xs, sess)
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
