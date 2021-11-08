import numpy as np
from numpy import arange
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor

from source.utility import DataSet


class Classifier:

    @staticmethod
    def nc_classify(dataSet: DataSet) -> np.ndarray:
        model = NearestCentroid()

        model = model.fit(dataSet.train_images, dataSet.train_labels)

        return model.predict(dataSet.test_images)

    @staticmethod
    def nsc_classify(dataSet: DataSet, K) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def nn_classify(dataSet: DataSet, neighbors) -> np.ndarray:
        model = KNeighborsClassifier(neighbors, weights='uniform')
        model.fit(dataSet.train_images, dataSet.train_labels)

        return model.predict(dataSet.test_images)

    def perceptron_bp_classify(dataSet: DataSet):
        model = SGDClassifier(loss='hinge', alpha=0, learning_rate='adaptive', eta0=0.1, max_iter=1000)
        model.fit(dataSet.train_images, dataSet.train_labels)

        return model.predict(dataSet.test_images)

    def perceptron_mse_classify(dataSet: DataSet):
        model = SGDClassifier(loss='squared_error', alpha=0, learning_rate='adaptive', eta0=0.1, max_iter=1000)

        model.fit(dataSet.train_images, dataSet.train_labels)

        return model.predict(dataSet.test_images)


