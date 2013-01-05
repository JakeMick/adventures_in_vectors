import numpy as np
from sklearn.svm import SVC
from matplotlib import pyplot as plt


def generate_xor(total_points=500, variance=0.1):
    """Generates XOR classification data with bimodal multivariate Gaussians
    for each class.

    Takes two parameters.

        total_points : how many data points for each Gaussian. 4x this number
            will be the total observations.

        variance : dispersion of the data points

    """
    mgc = lambda x, y: np.random.multivariate_normal(
        [x, y], [[variance, 0], [0, variance]], total_points)
    X_class_one = np.vstack((mgc(1, 1), mgc(-1, -1)))
    X_class_two = np.vstack((mgc(-1, 1), mgc(1, -1)))
    Y_class_one = np.tile(1, total_points * 2)
    Y_class_two = np.tile(0, total_points * 2)
    X = np.vstack((X_class_one, X_class_two))
    Y = np.hstack((Y_class_one, Y_class_two))
    return X, Y


def generate_svm_decision_info(X, Y):
    """Given data points and labels returns the SVM's decision function,
    along with the indices of the support vectors, x indices and y indices for
    plotting purposes.

    """
    model = SVC(C=1.0, kernel='linear', shrinking=True, probability=False,
                tol=0.001, cache_size=200, class_weight=None, verbose=False)
    model.fit(X, Y)
    xind, yind = np.meshgrid(np.linspace(X.min() - 1, X.max() + 1, 500),
                             np.linspace(X.min() - 1, X.max() + 1, 500))
    dec = model.decision_function(np.c_[xind.ravel(), yind.ravel()])
    dec = dec.reshape(xind.shape)
    return dec, model.support_, xind, yind


def go_through():
    X, Y = generate_xor()
    for i in xrange(2):
        decision, support_indices, xnew, ynew = \
            generate_svm_decision_info(X, Y)
        if i == 0:
            xind, yind = xnew, ynew
            plt.figure()
            plt.subplot(1,2,1)
        else:
            plt.subplot(1,2,2)
        decision_surface_plotter(X, Y, decision, support_indices, xind, yind)
        X = np.delete(X, support_indices, axis=0)
        Y = np.delete(Y, support_indices, axis=0)
    else:
        plt.show()


def decision_surface_plotter(X, Y, dec, support_indices, xind, yind):
    plt.imshow(dec, interpolation='nearest',
               extent=(xind.min(), xind.max(), yind.min(), yind.max()),
               aspect='auto', origin='lower', cmap='PiYG')
    plt.contour(xind, yind, dec, levels=[0], linewidths=2, linetypes='--')
    keepers = np.setdiff1d(np.arange(X.shape[0]), support_indices)
    plt.scatter(X[keepers, 0], X[keepers, 1], s=80, c=((Y[keepers] + .6) / 2),
                cmap='PiYG', marker='o', linewidths=1, alpha=.5,
                label="Out of Margin")
    plt.scatter(X[support_indices, 0], X[support_indices, 1], s=80,
                c=(Y[support_indices] * .3 + .1), cmap='PRGn', marker='d',
                linewidths=1, alpha=.4, label="Support Vectors")
    plt.axis([xind.min(), xind.max(), xind.min(), xind.max()])
    plt.legend(loc='best')

def main():
    go_through()

if __name__ == '__main__':
    main()
