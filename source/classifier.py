from collections import defaultdict

import numpy
import numpy as np
from matplotlib import pyplot as plt
from numpy import arange
from sklearn.cluster import KMeans
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor

from source.utility import DataSet
from mlxtend.plotting import plot_decision_regions


# Inspired by: https://github.com/SimplisticCode/ODA-ML/blob/master/NearestSubclassCentroid.py
class NearestSubclassCentroid:
    def __init__(self, K):
        self.K = K
        self.centroids = []

    def fit(self, X, y):
        subrows = defaultdict(list)
        for i in range(len(y)):
            # Collect indices of exemplars for the given class label
            subrows[y[i]].append(i)

        for index, label in enumerate(subrows.keys()):
            exemplars = X[subrows[label]]
            # compute centroid for exemplars
            subclasscentroids = self.__get_subclass_centroids(exemplars)
            for centroid in subclasscentroids:
                self.centroids.append({"centroid": centroid, "label": label})

        return self


    def __get_subclass_centroids(self, X):
        subclass_centroids = KMeans(n_clusters=self.K, random_state=0).fit(X).cluster_centers_
        return subclass_centroids

    def predict(self, X):
        results = []
        for sample in X:
            distances = []
            for centroid in self.centroids:
                distances.append((np.linalg.norm(sample - centroid["centroid"]), centroid["label"]))
            distances = sorted(distances, key=lambda x: x[0])
            results.append(distances[0][1])

        return numpy.array(results)

class Classifier:

    @staticmethod
    def nc_classify_train(dataSet: DataSet, shrink_treshold) -> NearestCentroid:
        model = NearestCentroid(shrink_threshold=shrink_treshold)
        model = model.fit(dataSet.train_images, dataSet.train_labels)

        return model

    @staticmethod
    def nsc_classify(dataSet: DataSet, K) -> NearestSubclassCentroid:
        model = NearestSubclassCentroid(K)
        model.fit(dataSet.train_images, dataSet.train_labels)

        return model


    @staticmethod
    def nn_classify(dataSet: DataSet, neighbors, weights) -> KNeighborsClassifier:
        model = KNeighborsClassifier(neighbors, weights=weights, n_jobs=-1)
        model.fit(dataSet.train_images, dataSet.train_labels)

        return model

    def perceptron_bp_classify(dataSet: DataSet, eta, learning_rate) -> SGDClassifier:
        model = SGDClassifier(loss='hinge', n_jobs=-1, eta0=eta, learning_rate=learning_rate)
        model.fit(dataSet.train_images, dataSet.train_labels)

        return model

    def perceptron_mse_classify(dataSet: DataSet, eta, learning_rate) -> SGDClassifier:
        model = SGDClassifier(loss='squared_error',
                              alpha=0,
                              learning_rate=learning_rate,
                              eta0=eta,
                              random_state=5,
                              n_jobs=-1
                              )
        model.fit(dataSet.train_images, dataSet.train_labels)

        return model


