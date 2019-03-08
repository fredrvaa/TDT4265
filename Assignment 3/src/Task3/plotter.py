import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("plots", exist_ok=True)

#Loss
plt.figure()

plt.title("Cross Entropy Loss")

task3_validation_loss = np.load('Task3/task3-np/Task3_validation_loss.npy')[::2]
task3_test_loss = np.load("Task3/task3-np/Task3_test_loss.npy")[::2]
task3_training_loss = np.load("Task3/task3-np/Task3_train_loss.npy")[::2]

plt.plot(task3_validation_loss, label='Validation Loss')
plt.plot(task3_test_loss, label='Test Loss')
plt.plot(task3_training_loss, label='Training Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.legend()

plt.savefig(os.path.join("plots", "Task3-loss.eps"))
plt.show()


#Accuracy
plt.figure()

plt.title("Accuracy")

task3_validation_accuracy = np.load('Task3/task3-np/Task3_validation_acc.npy')[::2]
task3_test_accuracy = np.load("Task3/task3-np/Task3_test_acc.npy")[::2]
task3_training_accuracy = np.load("Task3/task3-np/Task3_train_acc.npy")[::2]

plt.plot(task3_validation_accuracy, label='Validation accuracy')
plt.plot(task3_test_accuracy, label='Test accuracy')
plt.plot(task3_training_accuracy, label='Training accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend()

plt.savefig(os.path.join("plots", "Task3-acc.eps"))


plt.show()

print(task3_training_loss[-1])
print(task3_test_loss[-1])
print(task3_validation_loss[-1])
print('\n')
print(task3_training_accuracy[-1])
print(task3_test_accuracy[-1])
print(task3_validation_accuracy[-1])
