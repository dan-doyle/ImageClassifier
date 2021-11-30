from PIL import Image
import numpy as np
import torch

def process_image(path):
    
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    image = Image.open(path)
    
    # Resize keeping aspect ratio constant and giving smaller side 256 length
    width, height = image.size
    aspect_ratio = width / height
    if width > height:
        image = image.resize( (round(256 * aspect_ratio), 256))
    else:
        image = image.resize( (256, round(256 / aspect_ratio)))

    # Crop center 224x224 of image
    width, height = image.size
    new_width, new_height = 224, 224
    
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    image = image.crop((round(left), round(top), round(right), round(bottom)))
    
    # NumPy array and colour channels
    np_image = np.array(image) / 255
    
    # Normalise
    np_image = (np_image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    # Colour channel the first dimension
    np_image = np_image.transpose((2,0,1))
    # From NumPy to Tensor
    tensor_img = torch.from_numpy(np_image).type(torch.FloatTensor)
    
    return tensor_img