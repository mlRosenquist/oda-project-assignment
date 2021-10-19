import numpy as np
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor

from utility import DataSet


class Classifier:

    def nc_classify(dataSet: DataSet) -> np.ndarray:
        model = NearestCentroid()
        model = model.fit(dataSet.train_images, dataSet.train_labels)

        print(f"Training Set Score : {model.score(dataSet.train_images, dataSet.train_labels) * 100} %")
        return model.predict(dataSet.test_images)

    def nsc_classify(dataSet: DataSet, K) -> np.ndarray:
        raise NotImplementedError

    def nn_classify(dataSet: DataSet, neighbors):
        model = KNeighborsClassifier(neighbors, weights='uniform')
        model.fit(dataSet.train_images, dataSet.train_labels)

        print(f"Training Set Score : {model.score(dataSet.train_images, dataSet.train_labels) * 100} %")
        return model.predict(dataSet.test_images)

    def perceptron_bp_classify(dataSet: DataSet):
        model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

        model.fit(dataSet.train_images, dataSet.train_labels)

        print(f"Training Set Score : {model.score(dataSet.train_images, dataSet.train_labels) * 100} %")
        return model.predict(dataSet.test_images)

    def perceptron_mse_classify(dataSet: DataSet):
        model = MLPRegressor(random_state=1)

        model.fit(dataSet.train_images, dataSet.train_labels)

        print(f"Training Set Score : {model.score(dataSet.train_images, dataSet.train_labels) * 100} %")
        return model.predict(dataSet.test_images)


