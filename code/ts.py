import numpy as np
from matplotlib import pyplot as plt


def random_walk_bin(n=100000):
    np.random.seed(10)
    x = np.random.randint(0, 2, n)
    x[x == 0] = -1
    walk = np.cumsum(x)
    plt.plot(walk, 'k')
    plt.grid()
    plt.xlabel("Fixed time intervals")
    plt.ylabel("Heads - Tails")
    plt.title("Random Walk of Coin-Flipping")
    plt.show()


def random_walk_norm(n=100000):
    np.random.seed(10)
    x = np.random.normal(0, 1, n)
    walk = np.cumsum(x)
    plt.plot(walk, 'k')
    plt.grid()
    plt.xlabel("Fixed time intervals")
    plt.ylabel("Value")
    plt.title("Random Walk of Normal Distribution")
    plt.show()


def random_walk_dist():
    np.random.seed(10)
    w = np.cumsum(np.random.normal(0, 1, size=(1000, 1001)), axis=1)
    plt.plot(w[:50].T, alpha=.5)
    plt.title("50 Random Walks on a Normal Distribution")
    plt.xlabel("Fixed time intervals")
    plt.ylabel("Value")
    plt.grid()
    plt.plot(2.5 * np.sqrt(np.arange(1001)), 'k', linewidth=2)
    plt.plot(-2.5 * np.sqrt(np.arange(1001)), 'k', linewidth=2)
    plt.show()
    plt.title("Random Walk of Normal Distribution")
    plt.boxplot(w[:, ::100], bootstrap=1000)
    plt.grid()
    plt.xticks(np.arange(1, 12), np.arange(1001)[::100])
    plt.show()


class difference:
    def __init__(self, power=1):
        self.power = power

    def fit_transform(self, x):
        self.data = x
        self.difference = self.data.copy()
        for i in xrange(self.power):
            self.difference[1:] = self.difference[1:] - self.difference[:-1]
        return self.difference

    def inv_transform(self, x):
        for i in xrange(self.power):
            x = np.cumsum(x)
        return x


def differ_example():
    # Generate time series model
    np.random.seed(123)
    l = 1000
    const_drift = np.cumsum(np.array([1.0] * l))
    rw = np.cumsum(np.random.normal(0, 5, l))
    x = rw + const_drift
    # Plot the time series
    plt.plot(x, 'k')
    plt.xlabel("Fixed time intervals")
    plt.ylabel("Value")
    plt.grid()
    plt.title("Walk with constant drift")
    plt.show()
    # Plot the once differenced time series
    model1 = difference(power=1)
    res1 = model1.fit_transform(x)
    plt.plot(res1, 'k')
    plt.xlabel("Fixed time intervals")
    plt.ylabel("Once Differenced Value")
    plt.grid()
    plt.title("Delta'd Walk")
    plt.show()


def autocorrelation_fast(x):
    assert(len(x.shape) == 1)
    n = x.shape[0]
    x -= x.mean()
    trans = np.fft.fft(x, n=n * 2)
    acf = np.fft.ifft(trans * np.conjugate(trans))[:n]
    acf /= acf[0]
    return np.real(acf)


def autocorr_example():
    np.random.seed(3290)
    x = np.linspace(0, 4 * np.pi, 1000)
    y_trend = np.sin(x) + np.random.normal(0, 1, 1000)
    y_notrend = np.random.normal(0, 10, 1000)
    # No trend plotting
    plt.subplot(2, 1, 1)
    plt.plot(y_notrend, 'k')
    plt.title("Data without trend")
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.ylim(-1, 1)
    plt.plot(autocorrelation_fast(y_notrend), 'k')
    plt.title("Autocorrelation Function of Data without trend")
    plt.grid()
    plt.show()
    # Trend plotting
    plt.subplot(2, 1, 1)
    plt.plot(y_trend, 'k')
    plt.title("Data with trend")
    plt.grid()
    plt.subplot(2, 1, 2)
    plt.ylim(-1, 1)
    plt.plot(autocorrelation_fast(y_trend), 'k')
    plt.title("Autocorrelation Function of Data with trend")
    plt.grid()
    plt.show()


def lagmat(tseries, lag=2):
    input_shape = tseries.shape
    assert(len(input_shape) == 1)
    n = input_shape[0]
    values = np.concatenate((tseries[-1:0:-1], tseries))
    a, b = np.ogrid[lag:n, n - 1:n - lag - 2:-1]
    Tminus = values[a + b]
    return Tminus[:, 0], Tminus[:, 1:]


def main():
    #random_walk_bin()
    #random_walk_norm()
    #random_walk_dist()
    #differ_example()
    autocorr_example()

if __name__ == '__main__':
    main()
