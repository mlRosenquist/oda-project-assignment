import os
import time

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
    orl_dataSet_raw = Utility.load_ORL("data\\")
    orl_dataSet_2d = Utility.pca_transform(orl_dataSet_raw, 2)

    # defining parameter range
    param_grid = {
                  }

if __name__ == '__main__':
    for subClasses in [2, 3, 5]:
        # Visualization info
        figureTitle = "ORL - Nearest sub-class centroid"
        classifierName = "nearestSubClassCentroid"
        figurePrefix = f'pictures\\{classifierName}-{subClasses}'
        logPath = f'logs\\{classifierName}-{subClasses}-log.txt'
        logfile = open(logPath, 'w')

        # Load data sets
        orl_dataSet_raw = Utility.load_ORL("data\\")
        orl_dataSet_2d = Utility.pca_transform(orl_dataSet_raw, 2)

        start = time.time()
        results_raw = Classifier.nsc_classify(orl_dataSet_raw, subClasses)
        stop = time.time()
        print(f"Training and prediction time for {classifierName} : {stop - start}s", file=logfile)

        start = time.time()
        results_2d = Classifier.nsc_classify(orl_dataSet_2d, subClasses)
        stop = time.time()
        print(f"Training and prediction time for {classifierName} 2d: {stop - start}s", file=logfile)

        # Print Results
        print(classification_report(orl_dataSet_raw.test_labels, results_raw, digits=4), file=logfile)
        print(classification_report(orl_dataSet_2d.test_labels, results_2d, digits=4), file=logfile)

        # Visualize Confusion Matrix
        confplt = DataVisualization.ConfusionMatrix(orl_dataSet_raw.test_labels, results_raw, figureTitle + " (784D)")
        confplt.savefig(f'{figurePrefix}-confusion-784d.png')

        confplt = DataVisualization.ConfusionMatrix(orl_dataSet_2d.test_labels, results_2d, figureTitle + " (2D)")
        confplt.savefig(f'{figurePrefix}-confusion-2d.png')








