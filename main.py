import os

from sklearn import metrics
from sklearn.decomposition import PCA

from classifier import Classifier
from utility import Utility
import plotly.express as px



classifiers = ["nc", "nn"]
#classifiers = ["nc", "nsc", "nn", "perceptron_bp", "perceptron_mse"]

if __name__ == '__main__':
    # Data paths
    cwd = os.getcwd()
    mnist_folder = cwd + "\\MNIST"
    orl_folder = cwd + "\\ORL"
    orltxt_folder = cwd + "\\ORL_txt"

    # Load data sets
    mnist_dataSet = Utility.load_MNIST(mnist_folder)
    orl_dataSet = Utility.load_ORL(orltxt_folder)

    # MNIST data set full dimensions
    # Perform each classification
    mnist_results = []
    for classifier in classifiers:
        lbls = None
        if classifier == "nc":
            lbls = Classifier.nc_classify(mnist_dataSet)
        elif classifier == "nsc":
            lbls = Classifier.nsc_classify(mnist_dataSet, 1)
        elif classifier == "nn":
            lbls = Classifier.nn_classify(mnist_dataSet, 1)
        elif classifier == "perceptron_bp":
            lbls = Classifier.perceptron_bp_classify(mnist_dataSet)
        elif classifier == "perceptron_mse":
            lbls = Classifier.perceptron_mse_classify(mnist_dataSet)
        mnist_results.append(lbls)

    # Evaluate results
    for mnist_result in mnist_results:
        print('{:.2%}\n'.format(metrics.accuracy_score(mnist_dataSet.test_labels, mnist_result)))

        # Perform PCA
        pca = PCA(n_components=2)
        components = pca.fit_transform(mnist_dataSet.test_images)

        # Plot 2D
        fig = px.scatter(components, x=0, y=1, color=mnist_result)
        fig.show()

    # ORL data set full dimensions
    # Perform each classification
    orl_results_fullD = []
    for classifier in classifiers:
        lbls = None
        if classifier == "nc":
            lbls = Classifier.nc_classify(orl_dataSet)
        elif classifier == "nsc":
            lbls = Classifier.nsc_classify(orl_dataSet, 1)
        elif classifier == "nn":
            lbls = Classifier.nn_classify(orl_dataSet, 1)
        elif classifier == "perceptron_bp":
            lbls = Classifier.perceptron_bp_classify(orl_dataSet)
        elif classifier == "perceptron_mse":
            lbls = Classifier.perceptron_mse_classify(orl_dataSet)
        orl_results_fullD.append(lbls)

    # Evaluate results
    for orl_result in orl_results_fullD:
        print('{:.2%}\n'.format(metrics.accuracy_score(orl_dataSet.test_labels, orl_result)))

        # Perform PCA
        pca = PCA(n_components=2)
        components = pca.fit_transform(orl_dataSet.test_images)

        # Plot 2D
        fig = px.scatter(components, x=0, y=1, color=orl_result)
        fig.show()

    raise NotImplementedError


