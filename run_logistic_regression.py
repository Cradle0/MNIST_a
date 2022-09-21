from check_grad import check_grad
from utils import *
from logistic import *
from plot_digits import *

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


np.random.seed(16)

def run_logistic_regression():
    train_inputs, train_targets = load_train()
    # train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()

    N, M = train_inputs.shape

    #####################################################################
    # TODO:                                                             #
    # Set the hyperparameters for the learning rate, the number         #
    # of iterations, and the way in which you initialize the weights.   #
    #####################################################################

    # I used a learning rate of 0.05 for both small and large sets
    # I used 100, 500 iterations for the large and small sets respectively
    hyperparameters = {
        "learning_rate": 0.05,
        "weight_regularization": 0.,
        "num_iterations": 100
    }

    # This distribution worked the most consistent out of a binomial, normal
    # and uniform
    weights = np.random.randn(M+1,1) 
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    #####################################################################
    # TODO:                                                             #
    # Modify this section to perform gradient descent, create plots,    #
    # and compute test error.                                           #
    #####################################################################
    
    fig, ax = plt.subplots()

    for t in range(hyperparameters["num_iterations"]):
        f, df, y = logistic(weights, train_inputs, train_targets, hyperparameters)
        f_valid = logistic(weights, valid_inputs, valid_targets, hyperparameters)[0]
        weights = weights - hyperparameters["learning_rate"] * df
        ax.plot(t,f,'ro', t, f_valid, 'bs')
        
    # Get validation score to tune hyperparameters
    # print(evaluate(valid_targets, logistic_predict(weights, valid_inputs))[1])
    
    # part b
    # I obviously evaluated test after finalizing my hyperparameters.
    # test_inputs, test_targets = load_test()

    # print(evaluate(train_targets, logistic_predict(weights, train_inputs))[1])
    # print(evaluate(valid_targets, logistic_predict(weights, valid_inputs))[1])
    # print(evaluate(test_targets, logistic_predict(weights, test_inputs))[1])



    # part c 
    # Change the titles appropriately to whatever is loaded
    red = mpatches.Patch(color='red', label='Train')
    blue = mpatches.Patch(color='blue', label='Valid')

    plt.legend(handles = [red,blue])
    plt.title("MNIST_TRAIN")
    plt.savefig("plottrain")
    plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_check_grad(hyperparameters):
    """ Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,
                      weights,
                      0.001,
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == "__main__":
    run_logistic_regression()
    # run_pen_logistic_regression()
    