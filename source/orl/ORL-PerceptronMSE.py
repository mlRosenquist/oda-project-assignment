import os
import time

from matplotlib import pyplot as plt
from mlxtend.plotting import plot_decision_regions
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
    orl_dataSet_raw = Utility.load_ORL("data/")
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
    figurePrefix = f'pictures/{classifierName}'
    logPath = f'logs/{classifierName}-log.txt'
    logfile = open(logPath, 'w')

    # Load data sets
    orl_dataSet_raw = Utility.load_ORL("data/")
    orl_dataSet_2d = Utility.pca_transform(orl_dataSet_raw, 2)

    start = time.time()
    model_raw = Classifier.perceptron_mse_classify(orl_dataSet_raw, 0.001, 'adaptive')
    stop = time.time()
    print(f"Training time for {classifierName} : {stop - start}s", file=logfile)

    start = time.time()
    model_2d = Classifier.perceptron_mse_classify(orl_dataSet_2d, 1, 'invscaling')
    stop = time.time()
    print(f"Training time for {classifierName} 2d: {stop - start}s", file=logfile)

    # Print Results
    results_raw = model_raw.predict(orl_dataSet_raw.test_images)
    results_2d = model_2d.predict(orl_dataSet_2d.test_images)

    print(classification_report(orl_dataSet_raw.test_labels, results_raw, digits=4, zero_division=0), file=logfile,)
    print(classification_report(orl_dataSet_2d.test_labels, results_2d, digits=4, zero_division=0), file=logfile)

    # Visualize Confusion Matrix
    confplt = DataVisualization.ConfusionMatrix(orl_dataSet_raw.test_labels, results_raw, figureTitle + " (784D)")
    confplt.savefig(f"{figurePrefix}-confusion-784d.png")

    confplt = DataVisualization.ConfusionMatrix(orl_dataSet_2d.test_labels, results_2d, figureTitle + " (2D)")
    confplt.savefig(f"{figurePrefix}-confusion-2d.png")
    confplt.clf()

    # Visualize decision boundary
    ax = plot_decision_regions(orl_dataSet_2d.test_images, orl_dataSet_2d.test_labels, clf=model_2d, legend=0,
                               scatter_kwargs={'s': 10, 'edgecolor': None, 'alpha': 0.6})
    plt.title(f"{figureTitle} test (2D)")
    plt.xlabel("component 1")
    plt.ylabel("component 2")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])

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








