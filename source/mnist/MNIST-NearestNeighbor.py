import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from source.classifier import Classifier
from source.dataVisualization import DataVisualization
from source.utility import Utility

# 2D: {'n_neighbors': 5, 'weights': 'uniform'}
# FULLD: {'n_neighbors': 5, 'weights': 'distance'}
def tuneHyperParameters():
    # Load data sets
    mnist_dataSet_raw = Utility.load_MNIST("data\\")
    mnist_dataSet_2d = Utility.pca_transform(mnist_dataSet_raw, 2)

    # defining parameter range
    param_grid = {
        'n_neighbors': [2,3,5],
        'weights': ['uniform', 'distance']
                  }

    grid = GridSearchCV(KNeighborsClassifier(), param_grid, refit=True, n_jobs=-1)

    grid.fit(mnist_dataSet_raw.train_images, mnist_dataSet_raw.train_labels)
    print(grid.best_params_)

    grid = GridSearchCV(KNeighborsClassifier(), param_grid, refit=True, n_jobs=-1)

    grid.fit(mnist_dataSet_2d.train_images, mnist_dataSet_2d.train_labels)
    print(grid.best_params_)

if __name__ == '__main__':
    # Visualization info
    figureTitle = "MNIST - Nearest neighbor"
    classifierName = "nearestNeighbor"
    neighbors = 5
    figurePrefix = f'pictures\\{classifierName}-{neighbors}'
    logPath = f'logs\\{classifierName}-{neighbors}-log.txt'
    logfile = open(logPath, 'w')

    # Load data sets
    mnist_dataSet_raw = Utility.load_MNIST("data\\")
    mnist_dataSet_2d = Utility.pca_transform(mnist_dataSet_raw, 2)


    results_raw = Classifier.nn_classify(mnist_dataSet_raw, neighbors, 'uniform')
    results_2d = Classifier.nn_classify(mnist_dataSet_2d, neighbors, 'uniform')

    # Print Results
    print(classification_report(mnist_dataSet_raw.test_labels, results_raw), file=logfile)
    print(classification_report(mnist_dataSet_2d.test_labels, results_2d), file=logfile)

    # Visualize Confusion Matrix
    confplt = DataVisualization.ConfusionMatrix(mnist_dataSet_raw.test_labels, results_raw, f"{figureTitle} (784D)")
    confplt.savefig(f"{figurePrefix}-confusion-784d.png")

    confplt = DataVisualization.ConfusionMatrix(mnist_dataSet_2d.test_labels, results_2d, f"{figureTitle} (2D)")
    confplt.savefig(f"{figurePrefix}-confusion-2d.png")








