import os

from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from source.classifier import Classifier
from source.dataVisualization import DataVisualization
from source.utility import Utility

if __name__ == '__main__':
    # Visualization info
    figureTitle = "MNIST - Perceptron using BackPropagation"
    figurePath = "pictures\\"
    classifierName = "perceptronBP"
    filePrefix = figurePath+classifierName

    # Load data sets
    mnist_dataSet_raw = Utility.load_MNIST()
    mnist_dataSet_2d = Utility.pca_transform(mnist_dataSet_raw, 2)

    results_raw = Classifier.perceptron_bp_classify(mnist_dataSet_raw)
    results_2d = Classifier.perceptron_bp_classify(mnist_dataSet_2d)

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








