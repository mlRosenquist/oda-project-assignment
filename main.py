import os

from classifier import Classifier
from utility import Utility

if __name__ == '__main__':
    # Data paths
    cwd = os.getcwd()
    mnist_folder = cwd + "\\MNIST"
    orl_folder = cwd + "\\ORL"
    orltxt_folder = cwd + "\\ORL_txt"

    # Load data sets
    mnist_dataSet = Utility.load_MNIST(mnist_folder)
    #orl_dataSet = Utility.load_ORL(orltxt_folder)

    # MNIST

    ## 1. Nearest class centroid classifier
    #lbls = Classifier.nc_classify(mnist_dataSet)

    ## 2. Nearest sub-class centroid classifier using number of subclasses in the set {2,3,5}
    #lbls = Classifier.nsc_classify(mnist_dataSet, 1)

    ## 3. Nearest Neighbor classifier
    #lbls = Classifier.nn_classify(mnist_dataSet, 1)

    ## 4. Perceptron trained using Backpropagation
    #lbls = Classifier.perceptron_bp_classify(mnist_dataSet)

    ## 5. Perceptron trained using MSE (least squares solution)
    lbls = Classifier.perceptron_mse_classify(mnist_dataSet)


    # ORL

    ## 1. Nearest class centroid classifier

    ## 2. Nearest sub-class centroid classifier using number of subclasses in the set {2,3,5}

    ## 3. Nearest Neighbor classifier

    ## 4. Perceptron trained using Backpropagation

    ## 5. Perceptron trained using MSE (least squares solution)

