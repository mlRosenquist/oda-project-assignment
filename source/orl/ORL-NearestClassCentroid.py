import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestCentroid

from source.classifier import Classifier
from source.dataVisualization import DataVisualization
from source.utility import Utility

def tuneHyperParameters():
    # Load data sets
    orl_dataSet_raw = Utility.load_ORL()
    orl_dataSet_2d = Utility.pca_transform(orl_dataSet_raw, 2)

    # defining parameter range
    param_grid = {'shrink_threshold': np.arange(0.0, 1.0, 0.1)}

    grid = GridSearchCV(NearestCentroid(), param_grid, refit=True, verbose=3, n_jobs=-1)

    grid.fit(orl_dataSet_2d.train_images, orl_dataSet_2d.train_labels)
    print(grid.best_params_)
    grid_predictions = grid.predict(orl_dataSet_2d.test_images)
    print(classification_report(orl_dataSet_2d.test_labels, grid_predictions))

if __name__ == '__main__':
    # Visualization info
    figureTitle = "ORL - Nearest class centroid"
    figurePath = "pictures\\"
    classifierName = "nearestClassCentroid"
    filePrefix = figurePath+classifierName

    # Load data sets
    orl_dataSet_raw = Utility.load_ORL()
    orl_dataSet_2d = Utility.pca_transform(orl_dataSet_raw, 2)

    results_raw = Classifier.nc_classify(orl_dataSet_raw)
    results_2d = Classifier.nc_classify(orl_dataSet_2d)

    # Print Results
    print(classification_report(orl_dataSet_raw.test_labels, results_raw))
    print(classification_report(orl_dataSet_2d.test_labels, results_2d))

    # Visualize Scatter
    scatterplt = DataVisualization.ScatterPlot_2d(orl_dataSet_2d.test_images, orl_dataSet_2d.test_labels, 10, figureTitle + " (2D)")
    scatterplt.savefig(filePrefix+'-scatter-2d.png')

    # Visualize Confusion Matrix
    confplt = DataVisualization.ConfusionMatrix(orl_dataSet_raw.test_labels, results_raw, figureTitle + " (784D)")
    confplt.savefig(filePrefix+"-confusion-784d.png")

    confplt = DataVisualization.ConfusionMatrix(orl_dataSet_2d.test_labels, results_2d, figureTitle + " (2D)")
    confplt.savefig(filePrefix+"-confusion-2d.png")








