import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

from source.classifier import Classifier
from source.dataVisualization import DataVisualization
from source.utility import Utility

# 2D: {'alpha': 0, 'eta0': 0.35, 'learning_rate': 'adaptive', 'loss': 'hinge'}
# FULLD: {'alpha': 0, 'eta0': 0.00037, 'learning_rate': 'adaptive', 'loss': 'hinge'}
def tuneHyperParameters():
    # Load data sets
    orl_dataSet_raw = Utility.load_ORL()
    orl_dataSet_2d = Utility.pca_transform(orl_dataSet_raw, 2)

    # defining parameter range
    param_grid = {
        'alpha': [0],
        'loss': ['hinge'],
        'eta0': [0.00037, 0.00025, 0.0002],
        'learning_rate': ['constant', 'invscaling', 'adaptive']
                  }

    grid = GridSearchCV(SGDClassifier(), param_grid, refit=True, n_jobs=-1)

    grid.fit(orl_dataSet_raw.train_images, orl_dataSet_raw.train_labels)
    print(grid.best_params_)

    param_grid = {
        'alpha': [0],
        'loss': ['hinge'],
        'eta0': [0.45, 0.4, 0.35,],
        'learning_rate': ['constant', 'invscaling', 'adaptive']
    }

    grid = GridSearchCV(SGDClassifier(), param_grid, refit=True, n_jobs=-1)

    grid.fit(orl_dataSet_2d.train_images, orl_dataSet_2d.train_labels)
    print(grid.best_params_)

if __name__ == '__main__':
    # Visualization info
    figureTitle = "ORL - Perceptron using BackPropagation"
    figurePath = "pictures\\"
    classifierName = "perceptronBP"
    filePrefix = figurePath+classifierName

    # Load data sets
    orl_dataSet_raw = Utility.load_ORL()
    orl_dataSet_2d = Utility.pca_transform(orl_dataSet_raw, 2)

    results_raw = Classifier.perceptron_bp_classify(orl_dataSet_raw, 0.35, 'adaptive')
    results_2d = Classifier.perceptron_bp_classify(orl_dataSet_2d, 0.00037, 'adaptive')

    # Print Results
    print(classification_report(orl_dataSet_raw.test_labels, results_raw))
    print(classification_report(orl_dataSet_2d.test_labels, results_2d))

    # Visualize Scatter
    scatterplt = DataVisualization.ScatterPlot_2d(orl_dataSet_2d.test_images, orl_dataSet_2d.test_labels, 10, figureTitle + " (2D)")
    scatterplt.savefig(filePrefix+'-scatter-2d.png')

    # Visualize Confusion Matrix
    confplt = DataVisualization.ConfusionMatrix(orl_dataSet_raw.test_labels, results_raw, figureTitle + " (784D)")
    confplt.savefig(filePrefix+"-confusion-784d.png")

    confplt = DataVisualization.ConfusionMatrix(orl_dataSet_2d.test_labels, results_2d, figureTitle + " (2D)")
    confplt.savefig(filePrefix+"-confusion-2d.png")








