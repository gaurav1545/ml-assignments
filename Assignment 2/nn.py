import numpy as np
from data import get_train
from sklearn.metrics import confusion_matrix
import seaborn as sns


def sigmoid(x, derivative=False):
    sig_x = 1/(1+np.exp(-x))
    if derivative:
        # return sig_x*(1-sig_x)
        return x*(1-x)
    else:
        return sig_x


def ce(y_true, y_pred):
    return -(y_true/y_pred - (1-y_true)/(1-y_pred))


def initialize(units):
    '''
    units is a list, starting at the number of input nodes
    and ending with the number of output nodes.
    returns a list of 2d arrays.
    '''
    weights = []
    for i in range(len(units)-1):
        weights.append(np.random.normal(size=(units[i], units[i+1])))
    return weights


def get_error(y_true, y_pred):
    N = y_pred.shape[0]
    err = -np.sum(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))/N
    return err


def feed_forward(weights, inp):
    assert len(weights[0]) == inp.shape[-1]
    temp_inp = []
    outputs = []
    for i in inp:
        temp_inp.append(i)
    for layer in range(len(weights)):
        # outputs.append(sum(temp_inp)/len(temp_inp))
        outputs.append(temp_inp)
        for n in range(len(temp_inp)):
            temp_inp[n] = np.matmul(temp_inp[n], weights[layer])
            temp_inp[n] = sigmoid(temp_inp[n])
    pred = np.array(temp_inp).reshape(len(temp_inp))
    return outputs, pred


def backpropagate(weights, outputs, y_true, y_pred, alpha=0.01):
    update = np.zeros(weights[1].shape)
    for n in range(len(y_true)):
        update += outputs[1][n]*ce(y_true[n], y_pred[n])*sigmoid(y_pred[n], derivative=True)
    update /= len(y_true)
    weights[1] = weights[1] - alpha*update
    
    update = np.zeros(weights[0].shape)
    for n in range(len(y_true)):
        update += outputs[0][n]*ce(y_true[n], y_pred[n])*sigmoid(y_pred[n], derivative=True)
    update /= len(y_true)
    weights[0] = weights[0] - alpha*update
    
    return weights


def evaluate(labels, pred):
    pred = (pred > 0.5).astype(int)
    acc = (pred == labels).astype(int).sum()/len(pred)
    print(f'ACCURACY = {acc}')
    print('CONFUSION MATRIX:')
    print(confusion_matrix(labels, pred))


def plot(losses, title):
    epochs = [(i+1) for i in range(len(losses))]
    ax = sns.lineplot(x=epochs, y=losses)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    fig = ax.get_figure()
    fig.savefig(f'{title}.png')
    fig.clf()


def main():
    np.random.seed(0) # 0 42 50
    inp, labels = get_train(True)
    N = len(labels)
    train_inp, train_labels = inp[:N*4//5], labels[:N*4//5]
    test_inp, test_labels = inp[N*4//5:N], labels[N*4//5:N]
    for n_units in [30, 40, 50]:
        weights = initialize([2500, n_units, 1])
        # train_labels = np.zeros((2000))
        # weights = [np.ones((3,3)), np.ones((3,1))]
        # train_inp = np.ones((2,1,3))
        losses = []
        for i in range(150):
            out, train_pred = feed_forward(weights, train_inp)
            losses.append(get_error(train_labels, train_pred))
            weights = backpropagate(weights, out, train_labels, train_pred)
        print('***TRAIN***')
        evaluate(train_labels, train_pred)
        print('***TEST***')
        _, test_pred = feed_forward(weights, test_inp)
        evaluate(test_labels, test_pred)
        plot(losses, f'Hidden Units = {n_units}')


if __name__ == '__main__':
    main()