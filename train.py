import numpy as np
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt

from model import MLPGaussianRegressor


def main():

    parser = argparse.ArgumentParser()
    # Ensemble size
    parser.add_argument('--ensemble_size', type=int, default=10,
                        help='Size of the ensemble')
    # Maximum number of iterations
    parser.add_argument('--max_iter', type=int, default=5000,
                        help='Maximum number of iterations')
    # Batch size
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Size of batch')
    # Epsilon for adversarial input perturbation
    parser.add_argument('--epsilon', type=float, default=1e-2,
                        help='Epsilon for adversarial input perturbation')
    # Alpha for trade-off between likelihood score and adversarial score
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Trade off parameter for likelihood score and adversarial score')
    # Learning rate
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate for the optimization')
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
    np.random.seed(1)
    tf.set_random_seed(1)
    # Layer sizes
    sizes = [1, 20, 20, 2]
    # Input data
    xs = np.expand_dims(np.linspace(-5, 5, num=100, dtype=np.float32), -1)
    # Target data
    ys = np.cos(xs)

    ensemble = [MLPGaussianRegressor(args, sizes, 'model'+str(i)) for i in range(args.ensemble_size)]

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for itr in range(args.max_iter):
            for model in ensemble:
                indices = np.random.choice(np.arange(len(xs)), size=args.batch_size)
                x = xs[indices, :]
                y = ys[indices, :]

                feed = {model.input_data: x, model.target_data: y}
                _, nll, nll_gradients, mean, var = sess.run([model.train_op, model.nll, model.nll_gradients, model.mean, model.var], feed)

                if np.isnan(nll):
                    print mean, var
                    return

                if itr % 100 == 0:
                    print 'itr', itr, 'nll', nll

        test(ensemble, sess)


def test(ensemble, sess):
    test_xs = np.expand_dims(np.linspace(-10, 10, num=200, dtype=np.float32), -1)
    mean, var = ensemble_mean_var(ensemble, test_xs, sess)
    std = np.sqrt(var)
    upper = mean + 3*std
    lower = mean - 3*std

    plt.plot(test_xs, np.cos(test_xs), 'b-')
    plt.plot(test_xs, mean, 'r-')
    plt.plot(test_xs, upper, 'g-')
    plt.plot(test_xs, lower, 'c-')
    plt.show()

if __name__ == '__main__':
    main()
