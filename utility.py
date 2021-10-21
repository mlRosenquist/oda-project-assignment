from pydantic import BaseModel
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

    def load_MNIST(mnistFolder: str ) -> DataSet:

        dataSet: DataSet = DataSet()
        dataSet.train_images = StandardScaler().fit_transform(Utility.__loadMNISTImages(mnistFolder + "\\train-images.idx3-ubyte"))
        dataSet.test_images = StandardScaler().fit_transform(Utility.__loadMNISTImages(mnistFolder + "\\t10k-images.idx3-ubyte"))
        dataSet.train_labels = Utility.__loadMNISTLabels(mnistFolder + '\\train-labels.idx1-ubyte')
        dataSet.test_labels = Utility.__loadMNISTLabels(mnistFolder + '\\t10k-labels.idx1-ubyte')

        return dataSet

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

    def load_ORL(orlFolder: str ) -> DataSet:

        dataSet: DataSet = DataSet()

        all_images = Utility.__loadORLImages(orlFolder + "\\orl_data.txt")
        all_labels = Utility.__loadORLLabels(orlFolder + '\\orl_lbls.txt')

        train_images, test_images, train_labels, test_labels =\
            train_test_split(all_images, all_labels, test_size=0.3, random_state=42)

        dataSet.train_images = StandardScaler().fit_transform(train_images)
        dataSet.test_images = StandardScaler().fit_transform(test_images)
        dataSet.train_labels = train_labels
        dataSet.test_labels = test_labels
        return dataSet

    def __loadORLImages(path: str):
        dataframe = pd.read_csv(path, delimiter="\t", header=None)
        dataframe = dataframe.drop(columns=[400])
        return dataframe.T.to_numpy()


    def __loadORLLabels(path: str):
        dataframe = pd.read_csv(path, header=None)
        return dataframe.to_numpy().reshape(-1,)
