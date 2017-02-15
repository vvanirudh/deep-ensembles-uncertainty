import numpy as np
import ipdb


class DataLoader_RegressionToy():

    def __init__(self, args):

        self.xs = np.expand_dims(np.linspace(-2, 2, num=1000, dtype=np.float32), -1)

        self.ys = (self.xs**3) + np.random.normal(scale=1, size=self.xs.shape)

        # Standardize input features
        self.input_mean = np.mean(self.xs, 0)
        self.input_std = np.std(self.xs, 0)
        self.xs_standardized = (self.xs - self.input_mean)/self.input_std

        # Target mean and std
        self.target_mean = np.mean(self.ys, 0)[0]
        self.target_std = np.std(self.ys, 0)[0]

        self.batch_size = args.batch_size

    def next_batch(self):

        indices = np.random.choice(np.arange(len(self.xs_standardized)), size=self.batch_size)
        x = self.xs_standardized[indices, :]
        y = self.ys[indices, :]

        return x, y

    def get_data(self):

        return self.xs_standardized, self.ys

    def get_test_data(self):

        test_xs = np.expand_dims(np.linspace(-4, 4, num=2000, dtype=np.float32), -1)

        test_ys = (test_xs**3) + np.random.normal(scale=1, size=test_xs.shape)

        test_xs_standardized = (test_xs - self.input_mean)/self.input_std

        return test_xs_standardized, test_ys


class DataLoader_RegressionToy_withKink():

    def __init__(self, args):

        self.xs = np.expand_dims(np.linspace(-1, 1, num=1000, dtype=np.float32), -1)

        self.ys = np.zeros(shape=self.xs.shape)
        for i, t in enumerate(self.xs):
            if t > 0.25 or t < -0.25:
                self.ys[i] = 10*(t)**3 + np.random.normal(scale=.1)
            else:
                self.ys[i] = 30*np.sin(t) + np.random.normal(scale=.1)

        # Standardize input features
        self.input_mean = np.mean(self.xs, 0)
        self.input_std = np.std(self.xs, 0)
        self.xs_standardized = (self.xs - self.input_mean)/self.input_std

        # Target mean and std
        self.target_mean = np.mean(self.ys, 0)[0]
        self.target_std = np.std(self.ys, 0)[0]

        self.batch_size = args.batch_size

    def next_batch(self):
        indices = np.random.choice(np.arange(len(self.xs_standardized)), size=self.batch_size)
        x = self.xs_standardized[indices, :]
        y = self.ys[indices, :]

        return x, y

    def get_data(self):

        return self.xs_standardized, self.ys

    def get_test_data(self):

        test_xs = np.expand_dims(np.linspace(-1.5, 1.5, num=1000, dtype=np.float32), -1)

        test_ys = np.zeros(shape=test_xs.shape)
        for i, t in enumerate(test_xs):
            if t > 0.25 or t < -0.25:
                test_ys[i] = 10*(t)**3 + np.random.normal(scale=.1)
            else:
                test_ys[i] = 30*np.sin(t) + np.random.normal(scale=.1)

        test_xs_standardized = (test_xs - self.input_mean)/self.input_std

        return test_xs_standardized, test_ys


class DataLoader_RegressionToy_sinusoidal():

    def __init__(self, args):

        self.xs = np.expand_dims(np.linspace(-8, 8, num=1000, dtype=np.float32), -1)

        self.ys = 5*(np.sin(self.xs)) + np.random.normal(scale=1, size=self.xs.shape)

        # Standardize input features
        self.input_mean = np.mean(self.xs, 0)
        self.input_std = np.std(self.xs, 0)
        self.xs_standardized = (self.xs - self.input_mean)/self.input_std

        # Target mean and std
        self.target_mean = np.mean(self.ys, 0)[0]
        self.target_std = np.std(self.ys, 0)[0]

        self.batch_size = args.batch_size

    def next_batch(self):

        indices = np.random.choice(np.arange(len(self.xs_standardized)), size=self.batch_size)
        x = self.xs_standardized[indices, :]
        y = self.ys[indices, :]

        return x, y

    def get_data(self):

        return self.xs_standardized, self.ys

    def get_test_data(self):

        test_xs = np.expand_dims(np.linspace(-16, 16, num=2000, dtype=np.float32), -1)

        test_ys = 5*(np.sin(test_xs)) + np.random.normal(scale=1, size=test_xs.shape)

        test_xs_standardized = (test_xs - self.input_mean)/self.input_std

        return test_xs_standardized, test_ys


class DataLoader_RegressionToy_sinusoidal_break():

    def __init__(self, args):

        self.xs = np.expand_dims(np.linspace(-8, 8, num=1000, dtype=np.float32), -1)

        self.xs = np.expand_dims(np.delete(self.xs, np.arange(300, 700)), -1)

        self.ys = 5*(np.sin(self.xs)) + np.random.normal(scale=1, size=self.xs.shape)

        # Standardize input features
        self.input_mean = np.mean(self.xs, 0)
        self.input_std = np.std(self.xs, 0)
        self.xs_standardized = (self.xs - self.input_mean)/self.input_std

        # Target mean and std
        self.target_mean = np.mean(self.ys, 0)[0]
        self.target_std = np.std(self.ys, 0)[0]

        self.batch_size = args.batch_size

    def next_batch(self):

        indices = np.random.choice(np.arange(len(self.xs_standardized)), size=self.batch_size)
        x = self.xs_standardized[indices, :]
        y = self.ys[indices, :]

        return x, y

    def get_data(self):

        return self.xs_standardized, self.ys

    def get_test_data(self):

        test_xs = np.expand_dims(np.linspace(-8, 8, num=2000, dtype=np.float32), -1)

        test_ys = 5*(np.sin(test_xs)) + np.random.normal(scale=1, size=test_xs.shape)

        test_xs_standardized = (test_xs - self.input_mean)/self.input_std

        return test_xs_standardized, test_ys


class DataLoader_RegressionToy_break():

    def __init__(self, args):

        self.xs = np.expand_dims(np.linspace(-4, 4, num=100, dtype=np.float32), -1)

        self.xs = np.expand_dims(np.delete(self.xs, np.arange(40, 90)), -1)

        self.ys = 5*(self.xs**3) + np.random.normal(scale=1, size=self.xs.shape)

        # Standardize input features
        self.input_mean = np.mean(self.xs, 0)
        self.input_std = np.std(self.xs, 0)
        self.xs_standardized = (self.xs - self.input_mean)/self.input_std

        # Target mean and std
        self.target_mean = np.mean(self.ys, 0)[0]
        self.target_std = np.std(self.ys, 0)[0]

        self.batch_size = args.batch_size

    def next_batch(self):

        indices = np.random.choice(np.arange(len(self.xs_standardized)), size=self.batch_size)
        x = self.xs_standardized[indices, :]
        y = self.ys[indices, :]

        return x, y

    def get_data(self):

        return self.xs_standardized, self.ys

    def get_test_data(self):

        test_xs = np.expand_dims(np.linspace(-4, 4, num=200, dtype=np.float32), -1)

        test_ys = 5*(test_xs**3) + np.random.normal(scale=1, size=test_xs.shape)

        test_xs_standardized = (test_xs - self.input_mean)/self.input_std

        return test_xs_standardized, test_ys
