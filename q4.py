'''
Question 4 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt
import scipy
import matplotlib.image
from scipy import special


def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    for i in range(0,10):
        y = np.nonzero(train_labels == i)[0]
        means[i] = np.mean(train_data[y],axis = 0)
    return means 

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    covariances = np.zeros((10, 64, 64))
    I = np.identity(64)
    # Compute covariances
    for i in range(0,10):
        y = np.nonzero(train_labels == i)[0]
        mean = np.mean(train_data[y], axis = 0)
        x = train_data[y] - mean
        covariances[i] = np.dot(x.T,x) / len(y) + 0.01*I
    
    return covariances

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    gll = np.zeros(shape=(np.shape(digits)[0],10))


    # Calculating the inside of that exp is hard
    # Use the fact that covariance is PSD 
    # To get its eigenvalues and eigenvectors 

    for i in range(1,10):
        einval, einvec = np.linalg.eig(covariances[i])
        v_old = (digits - means[i])

        #(covariances[0] @ vec == val * vec)
        v_new = v_old @ einvec

        v_new = np.square(v_new)
        inexp = -1/2 * np.dot(v_new, np.diag(np.linalg.inv(np.diag(einval))))
        gll.T[i] = np.logaddexp(np.log(1/(np.sqrt(np.pi ** 64))), np.log(np.sqrt(1/(np.linalg.det(covariances[i]))))) + inexp

    return gll

def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    # undo the log 
    gll = np.exp(generative_likelihood(digits,means,covariances))

    cll = np.zeros(shape=(np.shape(digits)[0],10))

    cll = 1/10 * gll 
    
    cll = np.logaddexp(np.log(cll), -np.log(1/10*gll))

    return cll

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    # Compute as described above and return

    acll = 1/(np.shape(labels)[0]) * scipy.special.logsumexp(cond_likelihood)

    return acll

def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    
    # Compute and return the most likely class
    return np.argmax(cond_likelihood,axis=1)

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # Evaluation
    acll_test = avg_conditional_likelihood(test_data,test_labels,means,covariances)
    acll_train = avg_conditional_likelihood(train_data,train_labels,means,covariances)

    print("Average conditional likelihood of test data:", acll_test)
    print("Average conditional likelihood of training data:", acll_train)


    predict_test = classify_data(test_data,means,covariances)
    predict_train = classify_data(train_data,means,covariances)

    cll_train = conditional_likelihood(train_data, means, covariances)
    acc_train = np.count_nonzero(np.argmax(cll_train,axis=1) == (train_labels)) / (np.shape(train_labels)[0])
    print("Accuracy on training data:", acc_train)


    cll_test = conditional_likelihood(test_data, means, covariances)
    acc_test = np.count_nonzero(np.argmax(cll_test,axis=1) == (test_labels)) / (np.shape(test_labels)[0])
    print("Accuracy on training data:", acc_test)

    einval, einvec = np.linalg.eig(covariances)
    max_einval = np.amax(einval,axis=1)
    max_einval_indice = np.argmax(einval,axis=1)
    lead_einvec = np.zeros(shape=(10,64))


    for i in range(0,10):
        lead_einvec[i] = einvec[i][max_einval_indice[i]] 
        plt.imshow((lead_einvec[i].reshape(8,8)),cmap=matplotlib.cm.Greys)
        plt.savefig(str(i))


if __name__ == '__main__':
    main()
