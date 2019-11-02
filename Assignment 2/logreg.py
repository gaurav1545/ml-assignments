from data import get_train
from nn import *
import numpy as np


def train_logreg(weights, outputs, y_true, y_pred, alpha=0.01):
    update = np.zeros(weights[0].shape)
    for n in range(len(y_true)):
        update += -outputs[0][n]*ce(y_true[n], y_pred[n])*sigmoid(y_pred[n], derivative=True)
    update /= len(y_true)
    weights[0] = weights[0] - alpha*update
    
    return weights


def main():
    np.random.seed(0) #
    inp, labels = get_train(True)
    N = len(labels)
    train_inp, train_labels = inp[:N*4//5], labels[:N*4//5]
    test_inp, test_labels = inp[N*4//5:N], labels[N*4//5:N]
    weights = initialize([2500, 1])
    # train_labels = np.zeros((2000))
    # weights = [np.ones((3,3)), np.ones((3,1))]
    # train_inp = np.ones((2,1,3))
    losses = []
    for i in range(10):
        out, train_pred = feed_forward(weights, train_inp)
        losses.append(get_error(train_labels, train_pred))
        weights = train_logreg(weights, out, train_labels, train_pred)
    print('***TRAIN***')
    evaluate(train_labels, train_pred)
    print('***TEST***')
    _, test_pred = feed_forward(weights, test_inp)
    evaluate(test_labels, test_pred)
    plot(losses, '0 hidden layers')


if __name__ == '__main__':
    main()
