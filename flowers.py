import glob 

# FLOWERS_LABELS is a dictionary mapping the flower categories to their 
# corresponding neuron in the output layer.
FLOWERS_LABELS = {
    "roses": 0,
    "daisy": 1,
    "dandelion": 2,
    "sunflowers": 3,
    "tulips": 4
}

def get_flower_paths(data_dir):
    """
    Returns a dictionary containing a list of all the paths of the flowers in 
    each category.
    """
    return {
        "roses": glob.glob(f'{data_dir}/roses/*', recursive=True),
        "daisy": glob.glob(f'{data_dir}/daisy/*', recursive=True),
        "dandelion": glob.glob(f'{data_dir}/dandelion/*', recursive=True),
        "sunflowers": glob.glob(f'{data_dir}/sunflowers/*', recursive=True),
        "tulips": glob.glob(f'{data_dir}/tulips/*', recursive=True)
    }


