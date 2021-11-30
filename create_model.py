import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


def create_model(arch, device, hidden=512, output=102, dropout=0.25, learning_rate=0.002):
    """ 
    Take a pretrained model integrate a custom classifier layer 
    
    Parameters:
    arch (string): Architecture of pretrained network, either resnet or vgg 
    hidden (int): Number of units in the hidden layer of the classifier
    dropout (float): Probability indicating dropout rate to be used in model
    learning_rate (float): Learning rate of model
    
    Outputs:
    model
    criterion
    optimizer
    
    """
    # Initialise resnet pretrained model
    if arch == 'resnet':
        model = models.resnet18(pretrained=True)
        
        # Freeze parameters by turning off gradients 
        for p in model.parameters():
            p.requires_grad = False

        # New Feedforward classifier 
        classifier = nn.Sequential(nn.Linear(512,hidden),
                                  nn.ReLU(),
                                  nn.Dropout(p=dropout),
                                  nn.Linear(hidden,output),
                                  nn.LogSoftmax(dim=1))
        model.fc = classifier
        optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)


    
    elif arch == 'vgg':
        model = models.vgg16(pretrained=True)
        
        # Freeze parameters by turning off gradients 
        for p in model.parameters():
            p.requires_grad = False

        # New Feedforward classifier 
        model.classifier = nn.Sequential(nn.Linear(25088,hidden),
                                  nn.ReLU(),
                                  nn.Dropout(p=dropout),
                                  nn.Linear(hidden,output),
                                  nn.LogSoftmax(dim=1))
        
        optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)        

    criterion = nn.NLLLoss()
    
    model.to(device);
    
    print('\nSuccessfully initialised', arch, 'model')
        
    return model, criterion, optimizer
    