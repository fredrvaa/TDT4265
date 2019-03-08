import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("plots", exist_ok=True)

#Loss
plt.figure()

plt.title("Cross Entropy Loss")

resnet_validation_loss = np.load('Task3/task3-np/Task3_validation_loss.npy')[::2]
resnet_training_loss = np.load("Task3/task3-np/Task3_train_loss.npy")[::2]

best_validation_loss = np.load('Task2/net2-np/Task2-net2_validation_loss.npy')[::2]
best_training_loss = np.load("Task2/net2-np/Task2-net2_train_loss.npy")[::2]

plt.plot(resnet_validation_loss, label='ResNet - Validation Loss')
plt.plot(resnet_training_loss, label='ResNet - Training Loss')
plt.plot(best_validation_loss, label='Network 2 - Validation Loss')
plt.plot(best_training_loss, label='Network 2 - Training Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.legend()

plt.savefig(os.path.join("plots", "Task3-compare-loss.eps"))
plt.show()
