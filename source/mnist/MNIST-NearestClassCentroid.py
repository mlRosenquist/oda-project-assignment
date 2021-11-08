import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestCentroid

from source.classifier import Classifier
from source.dataVisualization import DataVisualization
from source.utility import Utility

# {'shrink_threshold': 0.01}
def tuneHyperParameters():
    # Load data sets
    mnist_dataSet_raw = Utility.load_MNIST()
    mnist_dataSet_2d = Utility.pca_transform(mnist_dataSet_raw, 2)

    # defining parameter range
    param_grid = {'shrink_threshold': [0, 0.025, 0.01, 0.005]}

    grid = GridSearchCV(NearestCentroid(), param_grid, refit=True, n_jobs=-1)

    grid.fit(mnist_dataSet_raw.train_images, mnist_dataSet_raw.train_labels)
    print(grid.best_params_)

    param_grid = {'shrink_threshold': [0, 0.01,0.005, 0.0025]}

    grid = GridSearchCV(NearestCentroid(), param_grid, refit=True, n_jobs=-1)

    grid.fit(mnist_dataSet_2d.train_images, mnist_dataSet_2d.train_labels)
    print(grid.best_params_)

tuneHyperParameters()
if __name__ == '__main__':
    # Visualization info
    figureTitle = "MNIST - Nearest class centroid"
    figurePath = "pictures\\"
    classifierName = "nearestClassCentroid"
    filePrefix = figurePath+classifierName

    # Load data sets
    mnist_dataSet_raw = Utility.load_MNIST()
    mnist_dataSet_2d = Utility.pca_transform(mnist_dataSet_raw, 2)

    results_raw = Classifier.nc_classify(mnist_dataSet_raw, 0.01)
    results_2d = Classifier.nc_classify(mnist_dataSet_2d, 0.01)

    # Print Results
    print(classification_report(mnist_dataSet_raw.test_labels, results_raw))
    print(classification_report(mnist_dataSet_2d.test_labels, results_2d))

    # Visualize Scatter
    scatterplt = DataVisualization.ScatterPlot_2d(mnist_dataSet_2d.test_images, mnist_dataSet_2d.test_labels, 10, figureTitle + " (2D)")
    scatterplt.savefig(filePrefix+'-scatter-2d.png')

    # Visualize Confusion Matrix
    confplt = DataVisualization.ConfusionMatrix(mnist_dataSet_raw.test_labels, results_raw, figureTitle + " (784D)")
    confplt.savefig(filePrefix+"-confusion-784d.png")

    confplt = DataVisualization.ConfusionMatrix(mnist_dataSet_2d.test_labels, results_2d, figureTitle + " (2D)")
    confplt.savefig(filePrefix+"-confusion-2d.png")








