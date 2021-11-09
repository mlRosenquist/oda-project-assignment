import os

from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from source.classifier import Classifier
from source.dataVisualization import DataVisualization
from source.utility import Utility

# 2D:
# FULLD:
def tuneHyperParameters():
    # Load data sets
    orl_dataSet_raw = Utility.load_ORL()
    orl_dataSet_2d = Utility.pca_transform(orl_dataSet_raw, 2)

    # defining parameter range
    param_grid = {
                  }

if __name__ == '__main__':
    # Visualization info
    figureTitle = "ORL - Nearest sub-class centroid"
    figurePath = "pictures\\"
    classifierName = "nearestSubClassCentroid"
    filePrefix = figurePath+classifierName

    # Load data sets
    orl_dataSet_raw = Utility.load_ORL()
    orl_dataSet_2d = Utility.pca_transform(orl_dataSet_raw, 2)

    results_raw = Classifier.nsc_classify(orl_dataSet_raw, 2)
    results_2d = Classifier.nsc_classify(orl_dataSet_2d, 2)

    # Print Results
    print(classification_report(orl_dataSet_raw.test_labels, results_raw))
    print(classification_report(orl_dataSet_2d.test_labels, results_2d))

    # Visualize Confusion Matrix
    confplt = DataVisualization.ConfusionMatrix(orl_dataSet_raw.test_labels, results_raw, figureTitle + " (784D)")
    confplt.savefig(filePrefix+"-confusion-784d.png")

    confplt = DataVisualization.ConfusionMatrix(orl_dataSet_2d.test_labels, results_2d, figureTitle + " (2D)")
    confplt.savefig(filePrefix+"-confusion-2d.png")








