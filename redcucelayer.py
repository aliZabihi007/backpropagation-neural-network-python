import numpy as np
import random
import sys
from sklearn import datasets
import progressbar
import time


class MLP:
    def __init__(self):
        data = datasets.load_iris()
        self.w1 = np.array(np.zeros((4, 3)))
        self.biase = np.array(np.zeros((3)))
        self.X = data.data
        self.y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.count = 0
        self.lastoutput = np.array(np.zeros((3, 3)))
        self.gradiantoutput = np.array(np.zeros((6)))

    def createdataarray(self):
        for i in range(0, 4, 1):
            for j in range(0, 3, 1):
                self.w1[i][j] = random.uniform(0, 1)

        for i in range(0, 3, 1):
            self.biase[i] = random.uniform(0, 1)

    def forwardingPass(self, indexX):

        self.hidden = np.dot(self.X[indexX], self.w1)
        result = [0, 0, 0]
        for i in range(0, 3, 1):
            result[i] = self.sigmoid(result[i] + self.biase[i])

        return result

    def evaluationMLP(self, output, indexY, teta):

        if (((((self.y[indexY][0] - output[0]) ** 2) / 2) + (((self.y[indexY][1] - output[1]) ** 2) / 2) + (
                ((self.y[indexY][2] - output[2]) ** 2) / 2)) < teta):

            self.count += 1
            self.lastoutput[indexY] = output
        else:

            self.count = 0

        return self.count

    def backpropagation(self, indexX, indexY, outy, learningrate):
        print(self.w1)
        for i in range(0, 3, 1):
            for j in range(0, 4, 1):
                self.w1[j][i] = self.w1[j][i] + (learningrate * (self.y[indexY][i] - outy[i]) * self.X[indexY][j])
            self.biase[i] = self.biase[i] + (1 * learningrate * (self.y[indexY][i] - outy[i]))
        print(self.w1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def showReasult(self, epack):
        sys.stdout.write("\033[1;31m")
        print("weight x TO hidden")
        print(self.w1)
        sys.stdout.write("\033[95m")
        sys.stdout.write("\033[94m")
        print("Out put")
        print(self.lastoutput)
        print("Y")
        print(self.y)
        sys.stdout.write("\033[0;32m")
        print("epack: ")
        print(int(epack / 5))


if __name__ == '__main__':
    mlp = MLP()
    mlp.createdataarray()
    index = 0
    epakc = 1
    iteration = 0
    itry = 0
    outer = 0
    print("wating ......")
    bar = progressbar.ProgressBar(maxval=3, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    start_time = time.time()
    while ((epakc / 150) < 7):
        if (itry == 3):
            itry = 0
        out = mlp.forwardingPass(index)
        itraion = mlp.evaluationMLP(out, itry, 0.5)
        mlp.backpropagation(index, itry, out, 0.9)
        index = index + 1
        if (index == 150):
            index = 0
        if (index % 50 == 0):
            itry += 1
        epakc += 1
        sys.stdout.write("wait ......\r%d%%" % int((epakc / 150)))

        sys.stdout.flush()
    mlp.showReasult(epack=epakc)
    print("--- %s seconds ---" % (time.time() - start_time))
# print(round(random.uniform(0, 1), 4))
