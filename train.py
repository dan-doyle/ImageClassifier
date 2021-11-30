import argparse
import json
import torch

from inputs import receive_args
from prepare_images import prepare_images
from create_model import create_model
from train_model import train_model
from validation import validation
from save_checkpoint import save_checkpoint
from category_to_label import category_to_label


def main():
    args = receive_args()
    
    print("Training using the parameters:\n", args)
    
    arch = args.arch
    output = args.output
    hidden = args.hidden
    dropout = args.dropout
    epochs = args.epochs
    learning_rate = args.learning
    
    save_dir = args.save_dir # To be used later with save_checkpoint as the folder to save in
    save_filename = args.checkpoint # filename when saved
    
    # Initialise loaders with prepare_images.py
    trainloader, validloader, testloader, train_image_datasets = prepare_images(args.dir, args.batchsize)
    
    # Set device to GPU or CPU
    device = 'cpu'
    if args.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print('\n Device set to ', str(device).upper())
    
    # Create model
    model, criterion, optimizer = create_model(arch, device, hidden, output, dropout, learning_rate)
    
    # Upload categories to labels file
    cat_to_name = category_to_label(args.categories)
    
    # Train model
    train_model(model, device, criterion, optimizer, epochs, trainloader, validloader)
    
    # Validation
    validation(model, device, criterion, testloader)
    
    # Save the checkpoint
    save_checkpoint(model, arch, hidden, output, epochs, learning_rate, optimizer, dropout, train_image_datasets, save_dir, save_filename)
    
    print('\n Model saved in checkpoints folder')
    
    
if __name__ == "__main__":
    main()
    