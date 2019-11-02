import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_data(x, y, i):
    x0, x1 = x[y == 0], x[y == 1]
    feature1_0 = np.ravel(x0[:, 1])
    feature2_0 = np.ravel(x0[:, 2])
    feature1_1 = np.ravel(x1[:, 1])
    feature2_1 = np.ravel(x1[:, 2])
    plt.scatter(feature1_0, feature2_0, color='r', marker='_')
    plt.scatter(feature1_1, feature2_1, color='g', marker='+')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.savefig(f'perceptron {i}/Dataset_{i}.jpg')
    plt.close(1)
    

def plot_line(x, y, w, i, k):
    feature1 = np.ravel(x[:, 1])
    x0, x1 = x[y == -1], x[y == 1]
    feature1_0 = np.ravel(x0[:, 1])
    feature2_0 = np.ravel(x0[:, 2])
    feature1_1 = np.ravel(x1[:, 1])
    feature2_1 = np.ravel(x1[:, 2])
    plt.scatter(feature1_0, feature2_0, color='r', marker='_')
    plt.scatter(feature1_1, feature2_1, color='g', marker='+')
    x_to_plot = feature1
    y_to_plot = (-w[1]/w[2])*feature1 - (w[0]/w[2])
    y_to_plot = np.ravel(y_to_plot)
    plt.plot(x_to_plot, y_to_plot, '-b')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.savefig(f'perceptron {i}/{i}_{k}.jpg')
    plt.close(1)


def perceptron(data, i):
    x = np.matrix(data.iloc[:, [4,1,2]])
    y = np.array(data.iloc[:, 3])
    plot_data(x, y, i)
    y[y == 0] = -1
    w = np.matrix(np.random.rand(3))
    w = w.T
    alpha = 1e-2
    for k in range(250):
        y_preds = np.sign(np.matmul(x,w)).ravel()
        misclassified = np.where(y_preds != y)[1]
        y_misclassified = np.matrix(y[misclassified])
        x_misclassified = x[misclassified, :]
        update = -y_misclassified*x_misclassified
        update = np.transpose(update)
        update *= alpha
        w -= update
        if (k+1) % 50 == 0:
            alpha /= 10
        plot_line(x, y, w, i, k)
    return np.equal(y, y_preds).astype(int).sum() / len(y)


def main():
    for i in range(1,4):
        data = pd.read_csv(f'data/dataset_{i}.csv', header=None)
        data = data.assign(bias = np.zeros(len(data)) + 1)
        acc = perceptron(data, i)
        print(acc)


if __name__ == '__main__':
    main()
