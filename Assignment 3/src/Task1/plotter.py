import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("plots", exist_ok=True)

#Loss
plt.figure()

plt.title("Cross Entropy Loss")

task1_validation_loss = np.load('Task1/task1-np/Task1_validation_loss.npy')
task1_test_loss = np.load("Task1/task1-np/Task1_test_loss.npy")
task1_training_loss = np.load("Task1/task1-np/Task1_train_loss.npy")

plt.plot(task1_validation_loss, label='Validation Loss')
plt.plot(task1_test_loss, label='Test Loss')
plt.plot(task1_training_loss, label='Training Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.legend()

plt.savefig(os.path.join("plots", "Task1-loss.eps"))
plt.show()


#Accuracy
plt.figure()

plt.title("Accuracy")

task1_validation_accuracy = np.load('Task1/task1-np/Task1_validation_acc.npy')
task1_test_accuracy = np.load("Task1/task1-np/Task1_test_acc.npy")
task1_training_accuracy = np.load("Task1/task1-np/Task1_train_acc.npy")

plt.plot(task1_validation_accuracy, label='Validation accuracy')
plt.plot(task1_test_accuracy, label='Test accuracy')
plt.plot(task1_training_accuracy, label='Training accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend()

plt.savefig(os.path.join("plots", "Task1-acc.eps"))


plt.show()

print(task1_validation_accuracy)
print(task1_test_accuracy)
print(task1_training_accuracy)

