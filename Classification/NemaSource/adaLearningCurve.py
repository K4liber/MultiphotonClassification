#!/usr/bin/env python3.6

import matplotlib.pyplot as plt

setSize = [1000, 10000, 100000, 1000000, 10000000]
trainLoss = [0.0, 0.0, 0.02, 3.74, 4.83]
testLoss = [9.33, 6.89, 6.57, 5.4, 5.01]
trainAccuracy = [100.0, 100.0, 99.95, 89.16, 86.02]
testAccuracy = [73.0, 80.05, 80.97, 84.35, 85.49]

plt.subplot(1, 2, 1)
plt.tight_layout(pad = 3.2)
plt.plot(setSize, trainLoss, label = "train loss")
plt.plot(setSize, testLoss, label = "test loss")
plt.xscale("log")
plt.xlabel("size of set")
plt.ylabel("log loss")
plt.title("ADA learning curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(setSize, trainAccuracy, label = "train accuracy")
plt.plot(setSize, testAccuracy, label = "test accuracy")
plt.xscale("log")
plt.xlabel("size of set")
plt.ylabel("accuracy [%]")
plt.title("ADA learning curve")
plt.legend()
plt.show()