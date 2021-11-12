import os

import numpy as np
import scipy.ndimage
from matplotlib import pyplot as plt
from scipy.ndimage import rotate
from sklearn.decomposition import PCA
from sklearn.metrics import ConfusionMatrixDisplay

from source.utility import Utility, DataSet


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
        plt.tight_layout()
        return plt

    @staticmethod
    def ConfusionMatrix(test_labels, result_labels, title) -> plt:
        disp = ConfusionMatrixDisplay.from_predictions(test_labels, result_labels)
        disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
        plt.title(title)
        plt.locator_params(nbins=10)
        return plt

if __name__ == '__main__':
    # MNIST

    mnist_folder = "mnist\\data\\"
    mnist_figurePath = "mnist\\pictures\\"

    # Load data sets
    mnist_dataSet_raw = Utility.load_MNIST(mnist_folder)
    pca = PCA(n_components=2)
    mnist_dataSet_2d = DataSet()
    mnist_dataSet_2d.train_images = pca.fit_transform(mnist_dataSet_raw.train_images)
    mnist_dataSet_2d.test_images = pca.transform(mnist_dataSet_raw.test_images)
    mnist_dataSet_2d.test_labels = mnist_dataSet_raw.test_labels
    mnist_dataSet_2d.train_labels = mnist_dataSet_raw.train_labels

    ## Scatter plot test data
    fig = DataVisualization.scatterPlot_2d(mnist_dataSet_2d.test_images, mnist_dataSet_2d.test_labels, "MNIST Test data", 'tab10')
    fig.savefig(f'{mnist_figurePath}mnist-scatter.png')
    plt.clf()

    ## Image before PCA
    num_row = 2
    num_col = 5  # plot images
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))
    for i in range(10):
        ax = axes[i // num_col, i % num_col]
        image = np.reshape(mnist_dataSet_raw.train_images[i], (28, 28))
        ax.imshow(image, cmap='gray')
        ax.set_title('Label: {}'.format(mnist_dataSet_raw.train_labels[i]))
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f'{mnist_figurePath}image-before-pca.png')
    plt.clf()

    ## Image after PCA
    # Reconstruct signal
    X_train_reconstructed = pca.inverse_transform(mnist_dataSet_2d.train_images)

    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))
    for i in range(10):
        ax = axes[i // num_col, i % num_col]
        image = np.reshape(X_train_reconstructed[i], (28, 28))
        ax.imshow(image, cmap='gray')
        ax.set_title('Label: {}'.format(mnist_dataSet_2d.train_labels[i]))
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(f'{mnist_figurePath}image-reconstructed-pca.png')
    plt.clf()

    # ORL
    orl_folder = "orl\\data\\"
    orl_figurePath = "orl\\pictures\\"

    # Load data sets
    orl_dataSet_original = Utility.load_ORL_original(orl_folder)
    orl_dataSet_original_2d = Utility.pca_transform(orl_dataSet_original, 2)
    orl_dataSet_raw = Utility.load_ORL(orl_folder)
    pca = PCA(n_components=2)
    orl_dataSet_2d = DataSet()
    orl_dataSet_2d.train_images = pca.fit_transform(orl_dataSet_raw.train_images)
    orl_dataSet_2d.test_images = pca.transform(orl_dataSet_raw.test_images)
    orl_dataSet_2d.test_labels = orl_dataSet_raw.test_labels
    orl_dataSet_2d.train_labels = orl_dataSet_raw.train_labels

    ## Scatter plot first 20 original data
    fig = DataVisualization.scatterPlot_2d(orl_dataSet_original_2d.test_images[0:200], orl_dataSet_original_2d.test_labels[0:200], "ORL first 20 original data", 'tab20')
    fig.savefig(f'{orl_figurePath}orl-scatter-original-first.png')
    fig.close()

    ## Scatter plot last 20 original data
    fig = DataVisualization.scatterPlot_2d(orl_dataSet_original_2d.test_images[201:400], orl_dataSet_original_2d.test_labels[201:400], "ORL last 20 original data", 'tab20')
    fig.savefig(f'{orl_figurePath}orl-scatter-original-second.png')
    fig.close()

    ## Scatter plot test data
    fig = DataVisualization.scatterPlot_2d(orl_dataSet_2d.test_images,
                                           orl_dataSet_2d.test_labels, "test", 'tab20')
    fig.savefig(f'{orl_figurePath}orl-scatter-test.png')
    fig.close()

    ## Image before PCA
    num_row = 2
    num_col = 3  # plot images
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))

    for i in range(6):
        ax = axes[i // num_col, i % num_col]
        image = np.reshape(orl_dataSet_raw.train_images[i], (30, 40))
        image = np.rot90(image)
        image = np.rot90(image)
        image = np.rot90(image)
        ax.imshow(image, cmap='gray')
        ax.set_title('Label: {}'.format(orl_dataSet_raw.train_labels[i]))
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f'{orl_figurePath}image-before-pca.png')
    plt.clf()

    ## Image after PCA
    # Reconstruct signal
    X_train_reconstructed = pca.inverse_transform(orl_dataSet_2d.train_images)

    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5 * num_col, 2 * num_row))

    for i in range(6):
        ax = axes[i // num_col, i % num_col]
        image = np.reshape(X_train_reconstructed[i], (30, 40))
        image = np.rot90(image)
        image = np.rot90(image)
        image = np.rot90(image)
        ax.imshow(image, cmap='gray')
        ax.set_title('Label: {}'.format(orl_dataSet_raw.train_labels[i]))
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(f'{orl_figurePath}image-reconstructed-pca.png')
    plt.clf()
