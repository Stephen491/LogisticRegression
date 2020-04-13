import numpy as np
from sigmoid import sigmoid

def getCost(x,y,theta, lambda1):
    m = np.size(x,0)
    h_0 = sigmoid(np.dot(x,theta))
    return (1/m)*np.sum(-y*np.log(h_0)-(1-y)*np.log(h_0)) + ((lambda1)/(2*m))*np.sum(np.square(np.vstack([[0],theta[1:]])))

def gradient(x,y,theta, lambda1):
    m = np.size(x,0)
    h_0 = sigmoid(np.dot(x, theta))
    return (1/m)*(np.sum(((h_0-y)*x),0)).reshape(np.size(theta,0),1) + (lambda1/m)*np.sum(np.vstack([[0],theta[1:]]))
