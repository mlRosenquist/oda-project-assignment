import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay


class DataVisualization:
    @staticmethod
    def ScatterPlot_2d(test_images, test_labels, numClasses, title) -> plt:
        plt.scatter(test_images[:, 0], test_images[:, 1],
                    c=test_labels, edgecolor='none', alpha=0.5,
                    cmap=plt.cm.get_cmap('tab10', numClasses))

        plt.xlabel('component 1')
        plt.ylabel('component 2')
        plt.colorbar()
        plt.title(title)
        return plt

    @staticmethod
    def ConfusionMatrix(test_labels, result_labels, title) -> plt:
        disp = ConfusionMatrixDisplay.from_predictions(test_labels, result_labels)
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
        plt.title(title)
        return plt