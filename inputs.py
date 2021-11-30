import argparse

def receive_args():
    parser = argparse.ArgumentParser(description='Receive model arguments')
                                     
    parser.add_argument('--dir', type = str, default = 'flowers', help = 'Flowers images path')
    parser.add_argument('--gpu', type = bool, default = True, help = 'GPU: True / False')
    parser.add_argument('--save_dir', type = str, default = 'checkpoints', help = 'Set directory to save checkpoints')
    parser.add_argument('--categories', type = str, default = 'cat_to_name.json', help = 'Categories to labels file') 
    parser.add_argument('--checkpoint', type = str, default = 'checkpoint.pth', help = 'Checkpoint filename')
   
                                     
    # Model related                              
    parser.add_argument('--arch', type = str, default = 'resnet', help = 'Pretrained model architecture: input resnet or vgg')
    parser.add_argument('--batchsize', type = int, default = 64, help = 'Batchsize')
    parser.add_argument('--hidden', type = int, default = 256, help = 'Number of hidden units')
    parser.add_argument('--output', type = int, default = 102, help = 'Number of outputs')
    parser.add_argument('--epochs', type = int, default = 3, help = 'Number of epochs')
    parser.add_argument('--learning', type = float, default = 0.002, help = 'Learning rate')
    parser.add_argument('--dropout', type = float, default = 0.25, help = 'dropout')
                                     
    return parser.parse_args()
