import numpy as np
from numpy import arange
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor

from source.utility import DataSet


class Classifier:

    @staticmethod
    def nc_classify(dataSet: DataSet, shrink_treshold) -> np.ndarray:
        model = NearestCentroid(shrink_threshold=shrink_treshold)

        model = model.fit(dataSet.train_images, dataSet.train_labels)

        return model.predict(dataSet.test_images)

    @staticmethod
    def nsc_classify(dataSet: DataSet, K) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def nn_classify(dataSet: DataSet, neighbors, weights) -> np.ndarray:
        model = KNeighborsClassifier(neighbors, weights=weights)
        model.fit(dataSet.train_images, dataSet.train_labels)

        return model.predict(dataSet.test_images)

    def perceptron_bp_classify(dataSet: DataSet, eta, learning_rate):
        model = SGDClassifier(loss='hinge', alpha=0, learning_rate=learning_rate, eta0=eta)
        model.fit(dataSet.train_images, dataSet.train_labels)

        return model.predict(dataSet.test_images)

    def perceptron_mse_classify(dataSet: DataSet, eta, learning_rate):
        model = SGDClassifier(loss='squared_error', alpha=0, learning_rate=learning_rate, eta0=eta, max_iter=1000)

        model.fit(dataSet.train_images, dataSet.train_labels)

        return model.predict(dataSet.test_images)


