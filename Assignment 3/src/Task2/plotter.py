import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs("plots", exist_ok=True)

#Loss
plt.figure()

plt.title("Cross Entropy Loss")

task2_net1_validation_loss = np.load('Task2/net1-np/Task2-net1_validation_loss.npy')[::2]
task2_net1_test_loss = np.load("Task2/net1-np/Task2-net1_test_loss.npy")[::2]
task2_net1_training_loss = np.load("Task2/net1-np/Task2-net1_train_loss.npy")[::2]

plt.plot(task2_net1_validation_loss, label='Validation Loss')
plt.plot(task2_net1_test_loss, label='Test Loss')
plt.plot(task2_net1_training_loss, label='Training Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.legend()

plt.savefig(os.path.join("plots", "Task2-net1-loss.eps"))
plt.show()


#Accuracy
plt.figure()

plt.title("Accuracy")

task2_net1_validation_accuracy = np.load('Task2/net1-np/Task2-net1_validation_acc.npy')[::2]
task2_net1_test_accuracy = np.load("Task2/net1-np/Task2-net1_test_acc.npy")[::2]
task2_net1_training_accuracy = np.load("Task2/net1-np/Task2-net1_train_acc.npy")[::2]

plt.plot(task2_net1_validation_accuracy, label='Validation accuracy')
plt.plot(task2_net1_test_accuracy, label='Test accuracy')
plt.plot(task2_net1_training_accuracy, label='Training accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend()

plt.savefig(os.path.join("plots", "Task2-net1-acc.eps"))


plt.show()

#Loss
plt.figure()

plt.title("Cross Entropy Loss")

task2_net2_validation_loss = np.load('Task2/net2-np/Task2-net2_validation_loss.npy')[::2]
task2_net2_test_loss = np.load("Task2/net2-np/Task2-net2_test_loss.npy")[::2]
task2_net2_training_loss = np.load("Task2/net2-np/Task2-net2_train_loss.npy")[::2]

plt.plot(task2_net2_validation_loss, label='Validation Loss')
plt.plot(task2_net2_test_loss, label='Test Loss')
plt.plot(task2_net2_training_loss, label='Training Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.legend()

plt.savefig(os.path.join("plots", "Task2-net2-loss.eps"))
plt.show()


#Accuracy
plt.figure()

plt.title("Accuracy")

task2_net2_validation_accuracy = np.load('Task2/net2-np/Task2-net2_validation_acc.npy')[::2]
task2_net2_test_accuracy = np.load("Task2/net2-np/Task2-net2_test_acc.npy")[::2]
task2_net2_training_accuracy = np.load("Task2/net2-np/Task2-net2_train_acc.npy")[::2]

plt.plot(task2_net2_validation_accuracy, label='Validation accuracy')
plt.plot(task2_net2_test_accuracy, label='Test accuracy')
plt.plot(task2_net2_training_accuracy, label='Training accuracy')

plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.legend()

plt.savefig(os.path.join("plots", "Task2-net2-acc.eps"))


plt.show()

print('Network1')
print(task2_net1_training_loss[-1])
print(task2_net1_test_loss[-1])
print(task2_net1_validation_loss[-1])
print('\n')
print(task2_net1_training_accuracy[-1])
print(task2_net1_test_accuracy[-1])
print(task2_net1_validation_accuracy[-1])

print('\nNetwork2')
print(task2_net2_training_loss[-1])
print(task2_net2_test_loss[-1])
print(task2_net2_validation_loss[-1])
print('\n')
print(task2_net2_training_accuracy[-1])
print(task2_net2_test_accuracy[-1])
print(task2_net2_validation_accuracy[-1])