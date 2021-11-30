import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def validation(model, device, criterion, loader):
    test_loss = 0
    test_accuracy = 0
    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            logps = model.forward(images)
            test_batch_loss = criterion(logps, labels)
            test_loss += test_batch_loss.item()

            # Accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test loss: {test_loss/len(loader):.3f}.. "
        f"Test accuracy: {test_accuracy/len(loader):.3f}")