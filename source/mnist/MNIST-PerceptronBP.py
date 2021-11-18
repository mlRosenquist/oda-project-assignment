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

# 2D: {'alpha': 0, 'eta0': 0.002, 'learning_rate': 'adaptive', 'loss': 'hinge'}
# FULLD: {'alpha': 0, 'eta0': 0.425, 'learning_rate': 'adaptive', 'loss': 'hinge'}
def tuneHyperParameters():
    # Load data sets
    mnist_dataSet_raw = Utility.load_MNIST("data/")
    mnist_dataSet_2d = Utility.pca_transform(mnist_dataSet_raw, 2)

    # defining parameter range
    param_grid = {
        'alpha': [0],
        'loss': ['hinge'],
        'eta0': [0.1, 0.01, 0.001, 0.0001, 0.00001],
        'learning_rate': ['adaptive'],
        'max_iter': [1000]
                  }

    #grid = GridSearchCV(SGDClassifier(), param_grid, refit=True, n_jobs=-1, verbose=3, )

    #grid.fit(mnist_dataSet_raw.train_images, mnist_dataSet_raw.train_labels)
    #print(grid.best_params_)

    param_grid = {
        'alpha': [0],
        'loss': ['hinge'],
        'eta0': [0.1, 0.01, 0.001, 0.0001, 0.00001],
        'learning_rate': ['adaptive', 'constant', 'invscaling'],
        'max_iter': [1000]
    }

    grid = GridSearchCV(SGDClassifier(), param_grid, refit=True, n_jobs=-1, verbose=3)

    grid.fit(mnist_dataSet_2d.train_images, mnist_dataSet_2d.train_labels)
    print(grid.best_params_)

if __name__ == '__main__':
    # Visualization info
    figureTitle = "MNIST - Perceptron using BackPropagation"
    classifierName = "perceptronBP"
    figurePrefix = f'pictures/{classifierName}'
    logPath = f'logs/{classifierName}-log.txt'
    logfile = open(logPath, 'w')

    # Load data sets
    mnist_dataSet_raw = Utility.load_MNIST("data/")
    mnist_dataSet_2d = Utility.pca_transform(mnist_dataSet_raw, 2)

    start = time.time()
    model_raw = Classifier.perceptron_bp_classify(mnist_dataSet_raw, 0.01, 'adaptive')
    stop = time.time()
    print(f"Training time for {classifierName} 2d: {stop - start}s", file=logfile)

    start = time.time()
    model_2d = Classifier.perceptron_bp_classify(mnist_dataSet_2d, 0.1, 'invscaling')
    stop = time.time()
    print(f"Training time for {classifierName} 2d: {stop - start}s", file=logfile)

    # Print Results
    results_raw = model_raw.predict(mnist_dataSet_raw.test_images)
    results_2d = model_2d.predict(mnist_dataSet_2d.test_images)

    print(classification_report(mnist_dataSet_raw.test_labels, results_raw, digits=4), file=logfile)
    print(classification_report(mnist_dataSet_2d.test_labels, results_2d, digits=4), file=logfile)

    # Visualize Confusion Matrix
    confplt = DataVisualization.ConfusionMatrix(mnist_dataSet_raw.test_labels, results_raw, f"{figureTitle} (784D)")
    confplt.savefig(f"{figurePrefix}-confusion-784d.png")
    confplt.clf()

    confplt = DataVisualization.ConfusionMatrix(mnist_dataSet_2d.test_labels, results_2d, f"{figureTitle} (2D)")
    confplt.savefig(f"{figurePrefix}-confusion-2d.png")
    confplt.clf()

    # Visualize decision boundary
    ax = plot_decision_regions(mnist_dataSet_2d.test_images, mnist_dataSet_2d.test_labels, clf=model_2d, legend=0, scatter_kwargs={'s':10, 'edgecolor': None, 'alpha': 0.2})
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

    ax = plot_decision_regions(mnist_dataSet_2d.train_images, mnist_dataSet_2d.train_labels, clf=model_2d, legend=0, scatter_kwargs={'s':10, 'edgecolor': None, 'alpha': 0.2})
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








