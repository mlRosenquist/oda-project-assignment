import os

from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from source.classifier import Classifier
from source.dataVisualization import DataVisualization
from source.utility import Utility

if __name__ == '__main__':
    # Visualization info
    figureTitle = "MNIST - Nearest neighbor"
    figurePath = "pictures\\"
    classifierName = "nearestNeighbor"
    filePrefix = figurePath+classifierName

    # Load data sets
    mnist_dataSet_raw = Utility.load_MNIST()
    mnist_dataSet_2d = Utility.pca_transform(mnist_dataSet_raw, 2)

    i = 5
    results_raw = Classifier.nn_classify(mnist_dataSet_raw, i)
    results_2d = Classifier.nn_classify(mnist_dataSet_2d, i)

    # Print Results
    print(classification_report(mnist_dataSet_raw.test_labels, results_raw))
    print(classification_report(mnist_dataSet_2d.test_labels, results_2d))

    # Visualize Scatter
    scatterplt = DataVisualization.ScatterPlot_2d(mnist_dataSet_2d.test_images, mnist_dataSet_2d.test_labels, 10, f"{figureTitle} (2D, n={i})")
    scatterplt.savefig(f"{filePrefix}-{i}-scatter-2d.png")

    # Visualize Confusion Matrix
    confplt = DataVisualization.ConfusionMatrix(mnist_dataSet_raw.test_labels, results_raw, f"{figureTitle} (784D, n={i})")
    confplt.savefig(f"{filePrefix}-{i}-confusion-784d.png")

    confplt = DataVisualization.ConfusionMatrix(mnist_dataSet_2d.test_labels, results_2d, f"{figureTitle} (2D, n={i})")
    confplt.savefig(f"{filePrefix}-{i}-confusion-2d.png")








