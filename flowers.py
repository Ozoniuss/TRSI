from enum import Enum
import glob

FLOWERS_LABELS = {
    "roses": 0,
    "daisy": 1,
    "dandelion": 2,
    "sunflowers": 3,
    "tulips": 4
}

def get_flower_paths(data_dir):
    return {
        "roses": glob.glob(f'{data_dir}/roses/*', recursive=True),
        "daisy": glob.glob(f'{data_dir}/daisy/*', recursive=True),
        "dandelion": glob.glob(f'{data_dir}/dandelion/*', recursive=True),
        "sunflowers": glob.glob(f'{data_dir}/sunflowers/*', recursive=True),
        "tulips": glob.glob(f'{data_dir}/tulips/*', recursive=True)
    }


