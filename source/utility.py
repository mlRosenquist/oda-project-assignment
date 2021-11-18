import os

from matplotlib import  pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from scipy.io import loadmat

class DataSet:
    train_images: np.ndarray
    test_images: np.ndarray
    train_labels: np.array
    test_labels: np.array


class Utility:

    @staticmethod
    def load_MNIST(path) -> DataSet:
        dataSet = loadmat(path+'mnist_loaded.mat')
        X_train = dataSet['train_images']
        X_train = np.reshape(X_train, (28, 28, 60000))
        X_train = X_train.transpose((1, 0, 2))

        X_train = np.reshape(X_train, (28 * 28, 60000))
        X_train = X_train.transpose()

        X_test = dataSet['test_images']
        X_test = np.reshape(X_test, (28, 28, 10000))
        X_test = X_test.transpose((1, 0, 2))

        X_test = np.reshape(X_test, (28 * 28, 10000))
        X_test = X_test.transpose()

        y_train = dataSet['train_labels'].reshape((-1,))
        y_test = dataSet['test_labels'].reshape((-1,))

        data = DataSet()
        data.train_images = X_train
        data.test_images = X_test
        data.train_labels = y_train
        data.test_labels = y_test
        return data

    @staticmethod
    def scale(dataSet: DataSet) -> DataSet:
        scaler = StandardScaler()

        dataSet.train_images = scaler.fit_transform(dataSet.train_images)
        dataSet.test_images = scaler.transform(dataSet.test_images)
        return dataSet

    @staticmethod
    def pca_transform(dataSet: DataSet, components: int) -> DataSet:
        pca_dataset: DataSet = DataSet()
        pca = PCA(n_components=components)

        pca_dataset.train_images = pca.fit_transform(dataSet.train_images)
        pca_dataset.test_images = pca.transform(dataSet.test_images)
        pca_dataset.test_labels = dataSet.test_labels
        pca_dataset.train_labels = dataSet.train_labels
        return pca_dataset


    @staticmethod
    def load_ORL(path) -> DataSet:
        dataSet: DataSet = DataSet()

        all_images = loadmat(path + 'orl_data.mat')['data'].transpose()
        all_labels = loadmat(path + 'orl_lbls.mat')['lbls'].reshape((-1,))

        train_images, test_images, train_labels, test_labels =\
            train_test_split(all_images, all_labels, test_size=0.3, random_state=3)

        dataSet.train_images = train_images
        dataSet.test_images = test_images
        dataSet.train_labels = train_labels
        dataSet.test_labels = test_labels
        return dataSet

    @staticmethod
    def load_ORL_original(path) -> DataSet:
        dataSet: DataSet = DataSet()

        all_images = loadmat(path + 'orl_data.mat')['data'].transpose()
        all_labels = loadmat(path + 'orl_lbls.mat')['lbls']

        dataSet.train_images = all_images
        dataSet.test_images = all_images
        dataSet.train_labels = all_labels
        dataSet.test_labels = all_labels
        return dataSet
