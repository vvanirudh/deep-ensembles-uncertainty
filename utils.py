import numpy as np


class DataLoader_RegressionToy():

    def __init__(self, args, infer=False):

        if not infer:
            self.xs = np.expand_dims(np.linspace(-1, 1, num=100, dtype=np.float32), -1)
        else:
            self.xs = np.expand_dims(np.linspace(-2, 2, num=200, dtype=np.float32), -1)

        self.ys = 5*(self.xs**3) + np.random.normal(scale=0.1, size=self.xs.shape)

        self.batch_size = args.batch_size

    def next_batch(self):

        indices = np.random.choice(np.arange(len(self.xs)), size=self.batch_size)
        x = self.xs[indices, :]
        y = self.ys[indices, :]

        return x, y

    def get_data(self):

        return self.xs, self.ys


class DataLoader_RegressionToy_withKink():

    def __init__(self, args, infer=False):

        if not infer:
            self.xs = np.expand_dims(np.linspace(-1, 1, num=100, dtype=np.float32), -1)
        else:
            self.xs = np.expand_dims(np.linspace(-2, 2, num=100, dtype=np.float32), -1)

        self.ys = np.zeros(shape=self.xs.shape)
        for i, t in enumerate(self.xs):
            if t > 0.25 or t < -0.25:
                self.ys[i] = 10*(t)**3 + np.random.normal(scale=0.1)
            else:
                self.ys[i] = 10*np.sin(t) + np.random.normal(scale=0.1)

        self.batch_size = args.batch_size

    def next_batch(self):
        indices = np.random.choice(np.arange(len(self.xs)), size=self.batch_size)
        x = self.xs[indices, :]
        y = self.ys[indices, :]

        return x, y

    def get_data(self):

        return self.xs, self.ys
