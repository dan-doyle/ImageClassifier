import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def save_checkpoint(model, arch, hidden, output, epochs, learning_rate, optimizer, dropout, train_image_datasets ,save_dir, save_filename):
    
    
    path = save_dir + '/' + save_filename
    
    checkpoint = {
              'model_arch': arch,
              'n_hidden_units': hidden,
              'n_outputs': output,
              'epochs': epochs,
              'learning_rate':learning_rate,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict,
              'dropout': dropout,
              'class_to_idx': train_image_datasets.class_to_idx
             }

    torch.save(checkpoint, path)