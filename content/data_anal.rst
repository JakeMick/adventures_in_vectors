Deleting Support Vectors: I made a stupid argument on the internet
##################################################################

:tags: visualization, data, python
:date: 2012-12-02
:category: data
:author: Jake Mick

##################
The question posed
##################
In r/MachineLearning

User kripaks asked

    Lets say I trained an SVM model on some training data. I then remove the
    support vectors from the training data and re-train the model. The new model
    will have new support vectors. But, what all has changed in this new model? Can
    I confidently say that the new model performs worse than the previous one?

Obviously this isn't the best idea. The support vectors define the max-margin
used by the SVM. Deleting them is censoring your most valuable evidence.

Now what actually happens to the hyperplane when you delete the support vectors?
The margin must get wider, that's a given. But how dramatically could you make
the angle change in $t+1$ from $t$ given the most pathological data for a linear kernel?

Turns out, not very much. I figured a multivariate Gaussian XOR
data set would be a perfect edge-case. It also isn't that ridiculous of a real
world data set. My thought was that the support vectors would be defined by
the smaller end of the Gaussian tail. Those would be deleted, and the hyperplane
would play Twister in front of my eyes. *I was definitely wrong.*

This is by no means a formal proof, but repeatedly running it couldn't get the
hyperplane to tilt.

.. image:: static/one_svm_xor.png
   :align: right

Most of the points are support vectors.

.. image:: static/two_svm_xor.png
   :align: right


.. image:: static/three_svm_xor.png
   :align: right

.. image:: static/four_svm_xor.png
   :align: right

.. image:: static/five_svm_xor.png

Repeating this over and over with randomly generated XOR data didn't change it.
Turns out that the soft margin is pretty robust to data censoring
in cases that are directly pathological to how the hyperplane is formed. It
digs a trough in the margin.

The code for those interested.

.. code-block:: python

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
        plt.scatter(X[keepers, 0], X[keepers, 1], s=80, c=Y[keepers],
                    cmap='PiYG', marker='o', linewidths=1, alpha=.5,
                    label="Out of Margin")
        plt.scatter(X[support_indices, 0], X[support_indices, 1], s=80,
                    c=Y[support_indices], cmap='PRGn', marker='d',
                    linewidths=1, alpha=.4, label="Support Vectors")
        plt.axis([xind.min(), xind.max(), xind.min(), xind.max()])
        plt.legend(loc='best')

    def main():
        go_through()

    if __name__ == '__main__':
        main()

-- JakeMick
