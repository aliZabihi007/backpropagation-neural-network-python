import numpy as np
import random
import sys
from sklearn import datasets
import progressbar
import time


#  تعریف کلاس به اسم mlp
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

    # تعریف تابعی برای مقدار دهی اولیه کردن به وزن ها و بایاس
    def createdataarray(self):
        for i in range(0, 4, 1):
            for j in range(0, 3, 1):
                self.w1[i][j] = random.uniform(0, 1)

        for i in range(0, 3, 1):
            for j in range(0, 3, 1):
                self.w2[i][j] = random.uniform(0, 1)

        for i in range(0, 6, 1):
            self.biase[i] = random.uniform(0, 1)

    # تعریف تابعی برای حرکت کردن به سمت جلو و ضرب و جمع مقاادیر وزن و بدست اوردن خروجی مطلوب
    def forwardingPass(self, indexX):
        self.hidden = np.dot(self.X[indexX], self.w1)

        for i in range(0, 3, 1):
            self.hidden[i] = self.sigmoid(self.hidden[i] + self.biase[i])

        result = (np.dot(self.hidden, self.w2))

        for i in range(0, 3, 1):
            result[i] = self.sigmoid(result[i] + self.biase[i + 3])

        return result

    #  ارزشیابی میکنیم که مقدار خروجی و مقدار جوابی که خودمان داریم چه جوابی بهینه خواهد بود
    def evaluationMLP(self, output, indexY, teta):

        if (((((self.y[indexY][0] - output[0]) ** 2) / 2) + (((self.y[indexY][1] - output[1]) ** 2) / 2) + (
                ((self.y[indexY][2] - output[2]) ** 2) / 2)) < teta):

            self.count += 1
            self.lastoutput[indexY] = output
        else:

            self.count = 0

        return self.count

    # آبدیت کردن وزن ها و اصلاح وزن ها و بدست اوردن مقادیر مطلوب
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

    # ابدیت کردن مقدار وزن
    def updateYtoHidden(self, eta):

        for i in range(0, 3, 1):
            for j in range(0, 3, 1):
                self.w2[j][i] = self.w2[j][i] + (eta * self.gradiantoutput[i] * self.hidden[j])
            self.biase[i + 3] = self.biase[i + 3] + (1 * self.gradiantoutput[i] * eta)

    # ابدیت کردن وزن در مرحله دوم
    def updateHiddenToX(self, eta, indexX):
        for i in range(0, 3, 1):
            for j in range(0, 4, 1):
                # print("eee",(eta * self.gradiantoutput[i + 3] * self.X[indexX][j]))
                self.w1[j][i] = self.w1[j][i] + (eta * self.gradiantoutput[i + 3] * self.X[indexX][j])
            self.biase[i] = self.biase[i] + (1 * self.gradiantoutput[i + 3] * eta)

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

    def mumentomcheck(self):
        pass

    # تابع شروع و اجراییی


if __name__ == '__main__':
    mlp = MLP()
    mlp.createdataarray()
    index = 0
    epakc = 1
    oneclass = mlp.X[0:50]
    classonecont = 0
    twoclass = mlp.X[50:100]
    classtwocont = 50
    threeclass = mlp.X[100:150]
    classthreecont = 100
    iteration = 0
    itry = 0
    rate = 0.9
    print("wait ......")
    # در این مورد در 300 ایپاک و مقادیر را به سه دسته تقسیم کرده و در هر مورد یکی را به شبکه میدهیم و وزن ها رو ابدیت میکنیم

    for ep in range(0, 300, 1):
        sys.stdout.write("wait ......\r%d%%" % int((ep / 300)))
        sys.stdout.flush()
        for i in range(0, 50, 1):
            for j in range(0, 3, 1):
                if (j == 0):

                    outputY = mlp.forwardingPass(classonecont + i)
                    iteration = mlp.evaluationMLP(outputY, j, 0.001)

                    # print("wating ......it :", iteration, "__", epakc / 150, "__in", index, "__i", itry, "__o", outer)
                    mlp.backpropagation(classonecont + i, j, outputY, rate)

                elif (j == 1):

                    outputY = mlp.forwardingPass(classtwocont + i)
                    iteration = mlp.evaluationMLP(outputY, j, 0.001)
                    # print("wating ......it :", iteration, "__", epakc / 150, "__in", index, "__i", itry, "__o", outer)
                    mlp.backpropagation(classtwocont + i, j, outputY, rate)
                elif (j == 2):
                    outputY = mlp.forwardingPass(classthreecont + i)
                    iteration = mlp.evaluationMLP(outputY, j, 0.001)
                    # print("wating ......it :", iteration, "__", epakc / 150, "__in", index, "__i", itry, "__o", outer)
                    mlp.backpropagation(classthreecont + i, j, outputY, rate)
            # اگر این کد زیر فعال شود مقادیر به صورت وقفی ابدیت خواهند شد و مقدار میگیرند و از  حالت ثایت خارج می شود
            # rate = (0.9) / (1 + (ep + 1)/(i+1))

    mlp.showReasult(epack=epakc)
