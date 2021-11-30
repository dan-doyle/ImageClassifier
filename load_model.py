import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from create_model import create_model

def load_model(filename, save_dir, device):
    
    checkpoint = torch.load(save_dir + '/' + filename)
    
    dropout = checkpoint['dropout'] 
    learning_rate = checkpoint['learning_rate']
    
    # Use create model to initialise a model with identical architecture
    model, criterion, optimizer = create_model(checkpoint['model_arch'],
                                              device,
                                              checkpoint['n_hidden_units'],
                                              checkpoint['n_outputs'],
                                              checkpoint['dropout'],
                                              checkpoint['learning_rate'],
                                              )
    # Load in weights and biases from checkpoint
    model.load_state_dict(checkpoint['state_dict'])

    class_to_idx = checkpoint['class_to_idx']
    
    return model, criterion, optimizer, class_to_idx 