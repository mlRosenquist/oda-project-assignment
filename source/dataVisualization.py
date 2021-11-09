import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import rotate
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay

from source.utility import Utility


class DataVisualization:
    @staticmethod
    def scatterPlot_2d(test_images, test_labels, title, cmap) -> plt:
        plt.scatter(test_images[:, 0], test_images[:, 1],
                    c=test_labels, edgecolor='none', alpha=0.5,
                    cmap=cmap)

        plt.xlabel('component 1')
        plt.ylabel('component 2')
        plt.colorbar()
        plt.title(title)
        return plt

    @staticmethod
    def confusion_Matrix(test_labels, result_labels, title) -> plt:
        disp = ConfusionMatrixDisplay.from_predictions(test_labels, result_labels)
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
        plt.title(title)
        return plt

if __name__ == '__main__':
    # MNIST

    mnist_folder = "mnist\\data\\"
    mnist_figurePath = "mnist\\pictures\\"

    # Load data sets
    mnist_dataSet_raw = Utility.load_MNIST(mnist_folder)
    mnist_dataSet_2d = Utility.pca_transform(mnist_dataSet_raw, 2)

    ## Scatter plot test data
    fig = DataVisualization.scatterPlot_2d(mnist_dataSet_2d.test_images, mnist_dataSet_2d.test_labels, "test", 'tab10')
    fig.savefig(f'{mnist_figurePath}mnist-scatter.png')
    fig.close()

    ## Image before PCA
    pixels = mnist_dataSet_raw.test_images[1]
    pixels = np.array(pixels, dtype='uint8')
    pixels = pixels.reshape((28, 28))
    pixels = rotate(pixels, 90)
    plt.title('Label is {label}'.format(label=1))
    plt.imshow(pixels, cmap='gray')
    plt.show()

    ## Image after PCA

    # ORL
    orl_folder = "orl\\data\\"
    orl_figurePath = "orl\\pictures\\"

    # Load data sets
    orl_dataSet_original = Utility.load_ORL_original(orl_folder)
    orl_dataSet_original_2d = Utility.pca_transform(orl_dataSet_original, 2)
    orl_dataSet_raw = Utility.load_ORL(orl_folder)
    orl_dataSet_2d = Utility.pca_transform(orl_dataSet_raw, 2)

    ## Scatter plot first 20 original data
    fig = DataVisualization.scatterPlot_2d(orl_dataSet_original_2d.test_images[0:60], orl_dataSet_original_2d.test_labels[0:60], "test", 'tab20')
    fig.savefig(f'{orl_figurePath}orl-scatter-original-first.png')
    fig.close()

    ## Scatter plot last 20 original data
    fig = DataVisualization.scatterPlot_2d(orl_dataSet_original_2d.test_images[61:120], orl_dataSet_original_2d.test_labels[61:120], "test", 'tab20')
    fig.savefig(f'{orl_figurePath}orl-scatter-original-second.png')
    fig.close()

    ## Scatter plot test data
    fig = DataVisualization.scatterPlot_2d(orl_dataSet_2d.test_images,
                                           orl_dataSet_2d.test_labels, "test", 'tab20')
    fig.savefig(f'{orl_figurePath}orl-scatter-test.png')
    fig.close()

    ## Image before PCA

    ## Image after PCA