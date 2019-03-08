import os
import matplotlib.pyplot as plt
import numpy as np

import Trainer
import Models


def main():
    trainer = Trainer.Trainer(Models.Model2(image_channels=3, num_classes=10, should_initialize_weights=True))
    trainer.train()

    os.makedirs("plots", exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(12, 8))
    plt.title("Cross Entropy Loss")
    plt.plot(trainer.VALIDATION_LOSS, label="Validation loss")
    plt.plot(trainer.TRAIN_LOSS, label="Training loss")
    plt.plot(trainer.TEST_LOSS, label="Testing Loss")
    plt.legend()
    plt.savefig(os.path.join("plots", "final_loss.png"))
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.title("Accuracy")
    plt.plot(trainer.VALIDATION_ACC, label="Validation Accuracy")
    plt.plot(trainer.TRAIN_ACC, label="Training Accuracy")
    plt.plot(trainer.TEST_ACC, label="Testing Accuracy")
    plt.legend()
    plt.savefig(os.path.join("plots", "final_accuracy.png"))
    plt.show()

    print("Final test accuracy:", trainer.TEST_ACC[-trainer.early_stop_count])
    print("Final validation accuracy:", trainer.VALIDATION_ACC[-trainer.early_stop_count])

    #Saving images as arrays
    os.makedirs("Task2-net2", exist_ok=True)
    np.save("Task2-net2/Task2-net2_validation_acc.npy", np.array(trainer.VALIDATION_ACC))
    np.save("Task2-net2/Task2-net2_test_acc.npy", np.array(trainer.TEST_ACC))
    np.save("Task2-net2/Task2-net2_train_acc.npy", np.array(trainer.TRAIN_ACC))

    np.save("Task2-net2/Task2-net2_validation_loss.npy", np.array(trainer.VALIDATION_LOSS))
    np.save("Task2-net2/Task2-net2_test_loss.npy", np.array(trainer.TEST_LOSS))
    np.save("Task2-net2/Task2-net2_train_loss.npy", np.array(trainer.TRAIN_LOSS))

if __name__ == "__main__":
    main()
