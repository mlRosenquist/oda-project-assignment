import os

from matplotlib import pyplot as plt
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
    mnist_dataSet_raw = Utility.load_MNIST()
    mnist_dataSet_2d = Utility.pca_transform(mnist_dataSet_raw, 2)

    # defining parameter range
    param_grid = {
        'alpha': [0],
        'loss': ['hinge'],
        'eta0': [0.00225, 0.002, 0.00175],
        'learning_rate': ['adaptive']
                  }

    grid = GridSearchCV(SGDClassifier(), param_grid, refit=True, n_jobs=-1)

    grid.fit(mnist_dataSet_raw.train_images, mnist_dataSet_raw.train_labels)
    print(grid.best_params_)

    param_grid = {
        'alpha': [0],
        'loss': ['hinge'],
        'eta0': [0.4327, 0.425, 0.4125],
        'learning_rate': ['adaptive']
    }

    grid = GridSearchCV(SGDClassifier(), param_grid, refit=True, n_jobs=-1)

    grid.fit(mnist_dataSet_2d.train_images, mnist_dataSet_2d.train_labels)
    print(grid.best_params_)

if __name__ == '__main__':
    # Visualization info
    figureTitle = "MNIST - Perceptron using BackPropagation"
    figurePath = "pictures\\"
    classifierName = "perceptronBP"
    filePrefix = figurePath+classifierName

    # Load data sets
    mnist_dataSet_raw = Utility.load_MNIST()
    mnist_dataSet_2d = Utility.pca_transform(mnist_dataSet_raw, 2)

    results_raw = Classifier.perceptron_bp_classify(mnist_dataSet_raw, 0.002, 'adaptive')
    results_2d = Classifier.perceptron_bp_classify(mnist_dataSet_2d, 0.425, 'adaptive')

    # Print Results
    print(classification_report(mnist_dataSet_raw.test_labels, results_raw))
    print(classification_report(mnist_dataSet_2d.test_labels, results_2d))

    # Visualize Scatter
    scatterplt = DataVisualization.ScatterPlot_2d(mnist_dataSet_2d.test_images, mnist_dataSet_2d.test_labels, 10, f"{figureTitle} (2D)")
    scatterplt.savefig(f"{filePrefix}-scatter-2d.png")

    # Visualize Confusion Matrix
    confplt = DataVisualization.ConfusionMatrix(mnist_dataSet_raw.test_labels, results_raw, f"{figureTitle} (784D)")
    confplt.savefig(f"{filePrefix}-confusion-784d.png")

    confplt = DataVisualization.ConfusionMatrix(mnist_dataSet_2d.test_labels, results_2d, f"{figureTitle} (2D)")
    confplt.savefig(f"{filePrefix}-confusion-2d.png")








