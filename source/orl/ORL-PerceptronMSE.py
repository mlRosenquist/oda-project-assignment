import os
import time

from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV

from source.classifier import Classifier
from source.dataVisualization import DataVisualization
from source.utility import Utility

# 2D: {'alpha': 0, 'eta0': 0.425, 'learning_rate': 'adaptive', 'loss': 'squared_error'}
# FULLD: {'alpha': 0, 'eta0': 0.001, 'learning_rate': 'invscaling', 'loss': 'squared_error'}
def tuneHyperParameters():
    # Load data sets
    orl_dataSet_raw = Utility.load_ORL("data\\")
    orl_dataSet_2d = Utility.pca_transform(orl_dataSet_raw, 2)

    # defining parameter range
    param_grid = {
        'alpha': [0],
        'loss': ['squared_error'],
        'eta0': [0.001],
        'learning_rate': ['adaptive']
                  }

    grid = GridSearchCV(SGDClassifier(), param_grid, refit=True, n_jobs=-1, verbose=3)

    grid.fit(orl_dataSet_raw.train_images, orl_dataSet_raw.train_labels)
    print(grid.best_params_)

    param_grid = {
        'alpha': [0],
        'loss': ['squared_error'],
        'eta0': [1e3, 1e2, 1e1, 1e0, 1e-1,  1e-2, 1e-3, 1e-4],
        'learning_rate': ['adaptive']
                  }

    grid = GridSearchCV(SGDClassifier(), param_grid, refit=True, n_jobs=-1, verbose=3)

    grid.fit(orl_dataSet_2d.train_images, orl_dataSet_2d.train_labels)
    print(grid.best_params_)

if __name__ == '__main__':
    # Visualization info
    figureTitle = "ORL - Perceptron using Perceptron using MSE"
    classifierName = "perceptronMSE"
    figurePrefix = f'pictures\\{classifierName}'
    logPath = f'logs\\{classifierName}-log.txt'
    logfile = open(logPath, 'w')

    # Load data sets
    orl_dataSet_raw = Utility.load_ORL("data\\")
    orl_dataSet_2d = Utility.pca_transform(orl_dataSet_raw, 2)

    start = time.time()
    results_raw = Classifier.perceptron_mse_classify(orl_dataSet_raw, 0.001, 'adaptive')
    stop = time.time()
    print(f"Training and prediction time for {classifierName} : {stop - start}s", file=logfile)

    start = time.time()
    results_2d = Classifier.perceptron_mse_classify(orl_dataSet_2d, 1, 'invscaling')
    stop = time.time()
    print(f"Training and prediction time for {classifierName} 2d: {stop - start}s", file=logfile)

    # Print Results
    print(classification_report(orl_dataSet_2d.test_labels, results_2d, digits=4, zero_division=0))
    print(classification_report(orl_dataSet_raw.test_labels, results_raw, digits=4, zero_division=0), file=logfile,)
    print(classification_report(orl_dataSet_2d.test_labels, results_2d, digits=4, zero_division=0), file=logfile)

    # Visualize Confusion Matrix
    confplt = DataVisualization.ConfusionMatrix(orl_dataSet_raw.test_labels, results_raw, figureTitle + " (784D)")
    confplt.savefig(f"{figurePrefix}-confusion-784d.png")

    confplt = DataVisualization.ConfusionMatrix(orl_dataSet_2d.test_labels, results_2d, figureTitle + " (2D)")
    confplt.savefig(f"{figurePrefix}-confusion-2d.png")








