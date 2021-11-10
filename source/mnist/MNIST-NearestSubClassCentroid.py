import os

from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from source.classifier import Classifier
from source.dataVisualization import DataVisualization
from source.utility import Utility

if __name__ == '__main__':
    # Visualization info
    figureTitle = "MNIST - Nearest sub-class centroid"
    classifierName = "nearestSubClassCentroid"
    subClasses = 5
    figurePrefix = f'pictures\\{classifierName}-{subClasses}'
    logPath = f'logs\\{classifierName}-{subClasses}-log.txt'
    logfile = open(logPath, 'w')

    # Load data sets
    mnist_dataSet_raw = Utility.load_MNIST("data\\")
    mnist_dataSet_2d = Utility.pca_transform(mnist_dataSet_raw, 2)


    results_raw = Classifier.nsc_classify(mnist_dataSet_raw, subClasses)
    results_2d = Classifier.nsc_classify(mnist_dataSet_2d, subClasses)

    # Print Results
    print(classification_report(mnist_dataSet_raw.test_labels, results_raw), file=logfile)
    print(classification_report(mnist_dataSet_2d.test_labels, results_2d), file=logfile)

    # Visualize Confusion Matrix
    confplt = DataVisualization.ConfusionMatrix(mnist_dataSet_raw.test_labels, results_raw, figureTitle + " (784D)")
    confplt.savefig(f'{figurePrefix}-confusion-784d.png')

    confplt = DataVisualization.ConfusionMatrix(mnist_dataSet_2d.test_labels, results_2d, figureTitle + " (2D)")
    confplt.savefig(f'{figurePrefix}-confusion-2d.png')








