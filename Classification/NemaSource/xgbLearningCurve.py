#!/usr/bin/env python3.6

import matplotlib.pyplot as plt

setSize = [1000, 10000, 100000, 1000000, 10000000]
trainLoss = [0.0, 0.0, 0.12, 3.04, 4.68]
testLoss = [8.12, 6.56, 6.19, 5.35, 4.98]
trainAccuracy = [100.0, 100.0, 99.66, 91.19, 86.44]
testAccuracy = [76.50, 81.0, 82.07, 84.51, 85.59]

plt.subplot(1, 2, 1)
plt.tight_layout(pad = 3.2)
plt.plot(setSize, trainLoss, label = "train loss")
plt.plot(setSize, testLoss, label = "test loss")
plt.xscale("log")
plt.xlabel("size of set")
plt.ylabel("log loss")
plt.title("XGB learning curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(setSize, trainAccuracy, label = "train accuracy")
plt.plot(setSize, testAccuracy, label = "test accuracy")
plt.xscale("log")
plt.xlabel("size of set")
plt.ylabel("accuracy [%]")
plt.title("XGB learning curve")
plt.legend()
plt.show()