from collections import defaultdict

import numpy
import numpy as np
from numpy import arange
from sklearn.cluster import KMeans
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
        model = NearestSubclassCentroid(K)

        model.fit(dataSet.train_images, dataSet.train_labels)

        predicition = model.predict(dataSet.test_images)
        return numpy.array(predicition)


    @staticmethod
    def nn_classify(dataSet: DataSet, neighbors, weights) -> np.ndarray:
        model = KNeighborsClassifier(neighbors, weights=weights, n_jobs=-1)
        model.fit(dataSet.train_images, dataSet.train_labels)

        return model.predict(dataSet.test_images)

    def perceptron_bp_classify(dataSet: DataSet, eta, learning_rate, margin):
        model = SGDClassifier(loss='hinge', n_jobs=-1)
        hinge = model.loss_functions['hinge']
        model.loss_function_ = (hinge[0], margin)
        model.fit(dataSet.train_images, dataSet.train_labels)

        return model.predict(dataSet.test_images)

    def perceptron_mse_classify(dataSet: DataSet, eta, learning_rate, margin):
        model = SGDClassifier(loss='squared_error', alpha=0, learning_rate=learning_rate, eta0=eta, n_jobs=-1)
        hinge = model.loss_functions['hinge']
        model.loss_function_ = (hinge[0], margin)
        model.fit(dataSet.train_images, dataSet.train_labels)

        return model.predict(dataSet.test_images)


# Taken from: https://github.com/SimplisticCode/ODA-ML/blob/master/NearestSubclassCentroid.py
class NearestSubclassCentroid():
    def __init__(self, K):
        self.centroids = None
        self.K = K

    def fit(self, X, y):
        y = y.reshape((-1,))
        subrows = defaultdict(list)
        for i in range(len(y)):
            # Collect indices of exemplars for the given class label
            subrows[y[i]].append(i)

        centroids = []
        for index, label in enumerate(subrows.keys()):
            exemplars = X[subrows[label]]
            # compute centroid for exemplars
            subclasscentroids = self.subclasscentroid(exemplars, self.K)
            for centroid in subclasscentroids:
                centroids.append({"centroid": centroid, "label": label})
        self.centroids = centroids
        return self

    def subclasscentroid(self, X, Nsubclasses):
        subclasscentroids = KMeans(n_clusters=Nsubclasses, random_state=0).fit(X).cluster_centers_
        return subclasscentroids

    def predict(self, X):
        results = []
        for sample in X:
            distances = []
            for centroid in self.centroids:
                distances.append((np.linalg.norm(sample - centroid["centroid"]), centroid["label"]))
            distances = sorted(distances, key=lambda x: x[0])
            results.append(distances[0][1])

        return results