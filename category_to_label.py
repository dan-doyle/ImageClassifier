import json

def category_to_label(filename):
    with open(filename, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name
    