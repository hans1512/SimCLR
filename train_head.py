from classification_head import ClassificationHead
from utils.arguments import get_config, get_arguments
from utils.utils import path_from_config
import numpy as np


def load(path: str) -> (np.ndarray, np.ndarray):
    x = np.load(f'./embeddings/x{path}.npy')
    y = np.load(f'./embeddings/y{path}.npy')
    return x, y


def main():
    config = get_config(get_arguments())
    save_path = path_from_config(config)
    x, y = load(save_path)
    net = ClassificationHead(config)
    net.to('cuda')

    net.fit(x, y)

    net.save()

main()
