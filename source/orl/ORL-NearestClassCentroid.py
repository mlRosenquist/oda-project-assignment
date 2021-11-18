import os
import time

import numpy as np
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestCentroid

from source.classifier import Classifier
from source.dataVisualization import DataVisualization
from source.utility import Utility


def tuneHyperParameters():
    # Load data sets
    orl_dataSet_raw = Utility.load_ORL("data/")
    orl_dataSet_2d = Utility.pca_transform(orl_dataSet_raw, 2)

    # defining parameter range
    param_grid = {
        'shrink_threshold': [None, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            }

    grid = GridSearchCV(NearestCentroid(), param_grid, refit=True, verbose=3, n_jobs=-1)

    grid.fit(orl_dataSet_raw.train_images, orl_dataSet_raw.train_labels)
    print(grid.best_params_)
    grid_predictions = grid.predict(orl_dataSet_raw.test_images)
    print(classification_report(orl_dataSet_raw.test_labels, grid_predictions))

    grid = GridSearchCV(NearestCentroid(), param_grid, refit=True, n_jobs=-1)

    grid.fit(orl_dataSet_2d.train_images, orl_dataSet_2d.train_labels)
    print(grid.best_params_)
    grid_predictions = grid.predict(orl_dataSet_2d.test_images)
    print(classification_report(orl_dataSet_2d.test_labels, grid_predictions))

if __name__ == '__main__':
    # Visualization info
    figureTitle = "ORL - Nearest class centroid"
    classifierName = "nearestClassCentroid"
    figurePrefix = f'pictures/{classifierName}'
    logPath = f'logs/{classifierName}-log.txt'
    logfile = open(logPath, 'w')

    # Load data sets
    orl_dataSet_raw = Utility.load_ORL("data/")
    orl_dataSet_2d = Utility.pca_transform(orl_dataSet_raw, 2)

    start = time.time()
    model_raw = Classifier.nc_classify_train(orl_dataSet_raw, 0.1)
    stop = time.time()
    print(f"Training time for {classifierName} 2d: {stop - start}s", file=logfile)

    start = time.time()
    model_2d = Classifier.nc_classify_train(orl_dataSet_2d, None)
    stop = time.time()
    print(f"Training time for {classifierName} 2d: {stop - start}s", file=logfile)

    # Print Results
    results_raw = model_raw.predict(orl_dataSet_raw.test_images)
    results_2d = model_2d.predict(orl_dataSet_2d.test_images)

    print(classification_report(orl_dataSet_raw.test_labels, results_raw, digits=4), file=logfile)
    print(classification_report(orl_dataSet_2d.test_labels, results_2d, digits=4), file=logfile)

    # Visualize Confusion Matrix
    confplt = DataVisualization.ConfusionMatrix(orl_dataSet_raw.test_labels, results_raw, figureTitle + " (784D)")
    confplt.savefig(f'{figurePrefix}-confusion-784d.png')

    confplt = DataVisualization.ConfusionMatrix(orl_dataSet_2d.test_labels, results_2d, figureTitle + " (2D)")
    confplt.savefig(f'{figurePrefix}-confusion-2d.png')
    confplt.clf()

    # Visualize decision boundary
    ax = plot_decision_regions(orl_dataSet_2d.test_images, orl_dataSet_2d.test_labels, clf=model_2d, legend=0,
                               scatter_kwargs={'s': 10, 'edgecolor': None, 'alpha': 0.6})
    plt.title(f"{figureTitle} test (2D)")
    plt.xlabel("component 1")
    plt.ylabel("component 2")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.6, box.height])

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles,
              labels,
              framealpha=0.3, scatterpoints=1, ncol=3, loc='center left', markerscale=2, bbox_to_anchor=(1, 0.5))
    plt.savefig(f'{figurePrefix}-boundary-test.png')
    plt.clf()

    ax = plot_decision_regions(orl_dataSet_2d.train_images, orl_dataSet_2d.train_labels, clf=model_2d, legend=0,
                               scatter_kwargs={'s': 10, 'edgecolor': None, 'alpha': 0.6})
    plt.title(f"{figureTitle} train (2D)")
    plt.xlabel("component 1")
    plt.ylabel("component 2")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles,
              labels,
              framealpha=0.3, scatterpoints=1, ncol=3, loc='center left', markerscale=2, bbox_to_anchor=(1, 0.5))
    plt.savefig(f'{figurePrefix}-boundary-train.png')
    plt.clf()








