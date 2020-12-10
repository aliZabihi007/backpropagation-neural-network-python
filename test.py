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
        self.w2 = np.array(np.zeros((3, 3)))
        self.hidden = np.array(np.zeros((3)))
        self.biase = np.array(np.zeros((6)))
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
            for j in range(0, 3, 1):
                self.w2[i][j] = random.uniform(0, 1)

        for i in range(0, 6, 1):
            self.biase[i] = random.uniform(0, 1)

    def forwardingPass(self, indexX):
        self.hidden = np.dot(self.X[indexX], self.w1)

        for i in range(0, 3, 1):
            self.hidden[i] = self.sigmoid(self.hidden[i] + self.biase[i])

        result = (np.dot(self.hidden, self.w2))

        for i in range(0, 3, 1):
            result[i] = self.sigmoid(result[i] + self.biase[i + 3])

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

        self.gradiantoutput[0] = outy[0] * (1 - outy[0]) * (self.y[indexY][0] - outy[0])
        self.gradiantoutput[1] = outy[1] * (1 - outy[1]) * (self.y[indexY][1] - outy[1])
        self.gradiantoutput[2] = outy[2] * (1 - outy[2]) * (self.y[indexY][2] - outy[2])

        for i in range(1, 4, 1):
            self.gradiantoutput[i + 2] = self.hidden[i - 1] * (1 - self.hidden[i - 1]) * (
                    self.gradiantoutput[i - 1] * self.w2[0][i - 1] + self.gradiantoutput[i - 1] * self.w2[1][i - 1] +
                    self.gradiantoutput[i - 1] * self.w2[2][i - 1])

        self.updateYtoHidden(learningrate)
        self.updateHiddenToX(learningrate, indexX)

    #  print(self.w1)

    def updateYtoHidden(self, eta):

        for i in range(0, 3, 1):
            for j in range(0, 3, 1):
                self.w2[j][i] = self.w2[j][i] + (eta * self.gradiantoutput[i] * self.hidden[j])
            self.biase[i + 3] = self.biase[i + 3] + (1 * self.gradiantoutput[i] * eta)

    def updateHiddenToX(self, eta, indexX):
        for i in range(0, 3, 1):
            for j in range(0, 4, 1):
                # print("eee",(eta * self.gradiantoutput[i + 3] * self.X[indexX][j]))
                self.w1[j][i] = self.w1[j][i] + (eta * self.gradiantoutput[i + 3] * self.X[indexX][j])
            self.biase[i] = self.biase[i] + (1 * self.gradiantoutput[i + 3] * eta)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def showReasult(self, epack):
        sys.stdout.write("\033[1;31m")
        print("weight x TO hidden")
        print(self.w1)
        sys.stdout.write("\033[95m")
        print("weight hidden TO y")
        print(self.w2)
        sys.stdout.write("\033[94m")
        print("Out put")
        print(self.lastoutput)
        print("Y")
        print(self.y)
        sys.stdout.write("\033[0;32m")
        print("epack: ")
        print(int(epack / 150))

    def mumentomcheck(self):
        pass
if __name__ == '__main__':
    mlp = MLP()
    mlp.createdataarray()
    index = 0
    epakc = 1
    iteration = 0
    itry = 0
    outer = 0
    rate=0.9
    print("")

    while ((epakc/150) < 100):
        if (itry == 3):
            itry = 0
        outputY = mlp.forwardingPass(index)

        iteration = mlp.evaluationMLP(outputY, itry, 0.01)
        #print("wating ......it :", iteration, "__", epakc / 150, "__in", index, "__i", itry, "__o", outer)
        #mlp.backpropagation(index, itry, outputY, 0.9)
        print(rate)
        mlp.backpropagation(index, itry, outputY, rate)
        index = index + 1
        if (iteration == 50):
            outer += 1
        if (index == 150):
            index = 0
            rate=((epakc*rate)/(epakc+(epakc/150)))


        sys.stdout.write("wait ......\r%d%%" % int((epakc/150)))

        sys.stdout.flush()

        if (index % 50 == 0):
            itry += 1
        epakc += 1




    mlp.showReasult(epack=epakc)

# print(round(random.uniform(0, 1), 4))
