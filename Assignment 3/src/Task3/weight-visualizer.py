import os
from torchvision import utils

def visualize_weights(model):
    filters = model.conv1.weight.data[:10]
    torchvision.utils.save_image(filters, os.path.join("plots", "weights" + ".png"))