import os

from pydantic import BaseModel
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class DataSet:
    train_images: np.ndarray
    test_images: np.ndarray
    train_labels: np.array
    test_labels: np.array


class Utility:

    @staticmethod
    def load_MNIST() -> DataSet:
        cwd = os.getcwd()
        mnist_folder = cwd + "\\data"

        scaler = StandardScaler()
        dataSet: DataSet = DataSet()

        x_train = Utility.__loadMNISTImages(mnist_folder + "\\mnist-train-images.idx3-ubyte")
        x_test = Utility.__loadMNISTImages(mnist_folder + "\\mnist-test-images.idx3-ubyte")

        dataSet.train_images = scaler.fit_transform(x_train)
        dataSet.test_images = scaler.transform(x_test)
        dataSet.train_labels = Utility.__loadMNISTLabels(mnist_folder + '\\mnist-train-labels.idx1-ubyte')
        dataSet.test_labels = Utility.__loadMNISTLabels(mnist_folder + '\\mnist-test-labels.idx1-ubyte')

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
    def __loadMNISTImages(path: str) -> np.ndarray:
        fp = open(path, "rb")

        magic = int.from_bytes(fp.read(4), "big")
        assert magic == 2051

        numImages = int.from_bytes(fp.read(4), "big")
        numRows = int.from_bytes(fp.read(4), "big")
        numCols = int.from_bytes(fp.read(4), "big")

        images_bytes = fp.read()
        images_array = [x for x in images_bytes]

        x = np.reshape(images_array, (numCols, numRows, numImages), order="F")
        x = x.transpose((1, 0, 2))

        x = np.reshape(x, (numRows*numRows, numImages), order="F")
        x = x.transpose()

        return x

    @staticmethod
    def __loadMNISTLabels(path: str) -> np.array:
        fp = open(path, "rb")

        magic = int.from_bytes(fp.read(4), "big")
        assert magic == 2049

        numLabels = int.from_bytes(fp.read(4), "big")

        labels_bytes = fp.read()
        labels_array = [x for x in labels_bytes]
        assert len(labels_array) == numLabels

        fp.close()
        return np.array(labels_array)

    @staticmethod
    def load_ORL() -> DataSet:
        cwd = os.getcwd()
        orl_folder = cwd + "\\data"
        dataSet: DataSet = DataSet()

        all_images = Utility.__loadORLImages(orl_folder + "\\orl_data.txt")
        all_labels = Utility.__loadORLLabels(orl_folder + '\\orl_lbls.txt')

        train_images, test_images, train_labels, test_labels =\
            train_test_split(all_images, all_labels, test_size=0.3, random_state=42)

        scaler = StandardScaler()
        dataSet.train_images = scaler.fit_transform(train_images)
        dataSet.test_images = scaler.transform(test_images)
        dataSet.train_labels = train_labels
        dataSet.test_labels = test_labels
        return dataSet

    @staticmethod
    def __loadORLImages(path: str):
        dataframe = pd.read_csv(path, delimiter="\t", header=None)
        dataframe = dataframe.drop(columns=[400])
        return dataframe.T.to_numpy()

    @staticmethod
    def __loadORLLabels(path: str):
        dataframe = pd.read_csv(path, header=None)
        return dataframe.to_numpy().reshape(-1,)
