import os
import time

import numpy
import numpy as np
from matplotlib import pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from source.classifier import Classifier
from source.dataVisualization import DataVisualization
from source.utility import Utility

if __name__ == '__main__':
    for subClasses in [2, 3, 5]:
        # Visualization info
        figureTitle = "MNIST - Nearest sub-class centroid"
        classifierName = "nearestSubClassCentroid"
        figurePrefix = f'pictures/{classifierName}-{subClasses}'
        logPath = f'logs/{classifierName}-{subClasses}-log.txt'
        logfile = open(logPath, 'w')

        # Load data sets
        mnist_dataSet_raw = Utility.load_MNIST("data/")
        mnist_dataSet_2d = Utility.pca_transform(mnist_dataSet_raw, 2)

        start = time.time()
        model_raw = Classifier.nsc_classify(mnist_dataSet_raw, subClasses)
        stop = time.time()
        print(f"Training and time for {classifierName} 2d: {stop - start}s", file=logfile)

        start = time.time()
        model_2d = Classifier.nsc_classify(mnist_dataSet_2d, subClasses)
        stop = time.time()
        print(f"Training and time for {classifierName} 2d: {stop - start}s", file=logfile)

        # Print Results
        results_raw = model_raw.predict(mnist_dataSet_raw.test_images)
        results_2d = model_2d.predict(mnist_dataSet_2d.test_images)

        print(classification_report(mnist_dataSet_raw.test_labels, results_raw, digits=4), file=logfile)
        print(classification_report(mnist_dataSet_2d.test_labels, results_2d, digits=4), file=logfile)

        # Visualize Confusion Matrix
        confplt = DataVisualization.ConfusionMatrix(mnist_dataSet_raw.test_labels, results_raw, figureTitle + " (784D)")
        confplt.savefig(f'{figurePrefix}-confusion-784d.png')

        confplt = DataVisualization.ConfusionMatrix(mnist_dataSet_2d.test_labels, results_2d, figureTitle + " (2D)")
        confplt.savefig(f'{figurePrefix}-confusion-2d.png')
        confplt.clf()

        # Visualize decision boundary
        ax = plot_decision_regions(mnist_dataSet_2d.test_images, mnist_dataSet_2d.test_labels, clf=model_2d, legend=0,
                                   scatter_kwargs={'s': 10, 'edgecolor': None, 'alpha': 0.2})
        plt.title(f"{figureTitle} test (2D)")
        plt.xlabel("component 1")
        plt.ylabel("component 2")

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height])

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles,
                  labels,
                  framealpha=0.3, scatterpoints=1, ncol=3, loc='upper right', markerscale=2)
        plt.savefig(f'{figurePrefix}-boundary-test.png')
        plt.clf()

        ax = plot_decision_regions(mnist_dataSet_2d.train_images, mnist_dataSet_2d.train_labels, clf=model_2d, legend=0,
                                   scatter_kwargs={'s': 10, 'edgecolor': None, 'alpha': 0.2})
        plt.title(f"{figureTitle} train (2D)")
        plt.xlabel("component 1")
        plt.ylabel("component 2")

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width, box.height])

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles,
                  labels,
                  framealpha=0.3, scatterpoints=1, ncol=3, loc='upper right', markerscale=2)
        plt.savefig(f'{figurePrefix}-boundary-train.png')
        plt.clf()








