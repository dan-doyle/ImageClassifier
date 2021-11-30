import argparse

def input_predict():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image', type = str, default = 'image.jpg', help = 'Flowers image path')
    parser.add_argument('--gpu', type = bool, default = True, help = 'GPU: True / False')
    parser.add_argument('--checkpoint', type = str, default = 'checkpoint.pth', help = 'Checkpoint filename')
    parser.add_argument('--topk', type = int, default = 5, help = 'Returns top k classes')
    parser.add_argument('--save_dir', type = str, default = 'checkpoints', help = 'Checkpoints load directory')
    parser.add_argument('--categories', type = str, default = 'cat_to_name.json', help = 'File with each category to label')

    return parser.parse_args()