import numpy as np
import random
import progressbar
import sys
from time import sleep


#  تعریف کلاس به اسم mlp
class MLP:
    def __init__(self):
        self.w1 = np.array(np.zeros((4, 3)))
        self.w2 = np.array(np.zeros((3, 1)))
        self.hidden = np.array(np.zeros((3)))
        self.biase = np.array(np.zeros((4)))
        self.X = np.array([[1, 0, 0, 1], [0, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1]])
        self.y = np.array([0, 1, 1, 0, 1])
        self.count = 0
        self.lastoutput = np.array(np.zeros((5)))
        self.gradiantoutput = np.array(np.zeros((4)))

    # تعریف تابعی برای مقدار دهی اولیه کردن به وزن ها و بایاس
    def createdataarray(self):
        for i in range(0, 4, 1):
            for j in range(0, 3, 1):
                self.w1[i][j] = round(random.uniform(0, 1), 4)

        for i in range(0, 3, 1):
            self.w2[i][0] = round(random.uniform(0, 1), 4)

        for i in range(0, 4, 1):
            self.biase[i] = round(random.uniform(0, 1), 4)

    # تعریف تابعی برای حرکت کردن به سمت جلو و ضرب و جمع مقاادیر وزن و بدست اوردن خروجی مطلوب

    def forwardingPass(self, indexX):
        self.hidden = np.dot(self.X[indexX], self.w1)

        for i in range(0, 3, 1):
            self.hidden[i] = round(self.sigmoid(self.hidden[i] + self.biase[i]), 4)

        result = np.sum(np.dot(self.hidden, self.w2))
        return round(self.sigmoid(result + self.biase[3]), 4)

    #  ارزشیابی میکنیم که مقدار خروجی و مقدار جوابی که خودمان داریم چه جوابی بهینه خواهد بود
    def evaluationMLP(self, output, indexY, teta):
        if (((self.y[indexY] - output) ** 2) / 2 < teta):

            self.count += 1
            self.lastoutput[indexY] = output
        else:

            self.count = 0
        return self.count

    # آبدیت کردن وزن ها و اصلاح وزن ها و بدست اوردن مقادیر مطلوب
    def backpropagation(self, indexY, outy, learningrate):
        self.gradiantoutput[0] = round(outy * (1 - outy) * (self.y[indexY] - outy), 4)
        for i in range(1, 4, 1):
            self.gradiantoutput[i] = self.hidden[i - 1] * (1 - self.hidden[i - 1]) * (
                    self.gradiantoutput[0] * self.w2[i - 1])
        self.updateYtoHidden(learningrate)
        self.updateHiddenToX(learningrate, indexY)

    # ابدیت کردن مقدار وزن
    def updateYtoHidden(self, eta):

        for i in range(0, 3, 1):
            self.w2[i][0] = round(self.w2[i][0] + (eta * self.gradiantoutput[0] * self.hidden[i]), 4)
        self.biase[3] = round(self.biase[3] + (1 * self.gradiantoutput[0] * eta), 4)

    # ابدیت کردن وزن در مرحله دوم
    def updateHiddenToX(self, eta, indexX):
        for i in range(0, 3, 1):
            for j in range(0, 4, 1):
                self.w1[j][i] = round(self.w1[j][i] + (eta * self.gradiantoutput[i + 1] * self.X[indexX][j]), 4)
            self.biase[i] = round(self.biase[i] + (1 * self.gradiantoutput[i + 1] * eta), 4)

    # تعریف تابع سیگموید
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # نمایش دادن خروجی مطلوب
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


    # تابع شروع و اجراییی
if __name__ == '__main__':
    mlp = MLP()
    mlp.createdataarray()
    index = 0
    epakc = 1
    iteration = 0
    print("wating ......")
    # زمانی هر 5 مورد درست بود و خروجی مناسب بدهد پایان ابدیت شدن وزن ها را داریم
    while (iteration < 5):
        print("wating ......")
        outputY = mlp.forwardingPass(index)
        iteration = mlp.evaluationMLP(outputY, index, 0.0001)
        mlp.backpropagation(index, outputY, 0.9)
        index = index + 1
        if (index == 5):
            index = 0

        # //

        epakc += 1

    mlp.showReasult(epack=epakc)
    # print(round(random.uniform(0, 1), 4))
