import os
import matplotlib.pyplot as plt

import Trainer
import Model
import filter-visualizer
import weight-visualizer


def main():
    #Train model
    trainer = Trainer.ResNetTrainer(Model.ResNetModel)
    trainer.train()

    os.makedirs("plots", exist_ok=True)

    #Save plots and show them
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

    #Visualize filters and weights
    model = trainer.model.model
    filter-visualizer.visualize_filters(model)
    weight-visualizer.visualize_weights(model)

if __name__ == "__main__":
    main()
