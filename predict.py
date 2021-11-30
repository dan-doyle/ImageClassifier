import torch
import json

from input_predict import input_predict
from load_model import load_model
from process_image import process_image
from category_to_label import category_to_label

def main():
    
    args = input_predict()
    topk = args.topk
    image_file_path = args.image # To be used in process_image function
    
    device = 'cpu'
    if args.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cat_to_name = category_to_label(args.categories)
        
    model, criterion, optimizer, class_to_idx = load_model(args.checkpoint, args.save_dir, device)
    
    print('Successfully loaded checkpoint {}'.format(args.checkpoint))
    
    image_as_tensor = process_image(image_file_path)
    image_as_tensor = image_as_tensor.to(device)
    
    class_to_idx_dict = {v: k for k, v in class_to_idx.items()}
    
    model.eval()
    with torch.no_grad():
        image_as_tensor.unsqueeze_(0)
        log_ps = model.forward(image_as_tensor)
        ps = torch.exp(log_ps)

        top_ps, top_classes = ps.topk(topk)
        top_ps, top_classes = top_ps[0].tolist(), top_classes[0].tolist()
        
        named_top_classes = []
        for key in top_classes:
            named_top_classes.append(class_to_idx_dict[key])
        
        labels = [cat_to_name[i] for i in named_top_classes]
        
        print('\n')
        for i in range(len(top_classes)):
            print('{}. {} with probability {:.1f}%'.format(i+1, labels[i], top_ps[i]*100))    
        
                  
if __name__ == "__main__":
    main()