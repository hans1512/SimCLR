import numpy as np

def load(path: str) -> (np.ndarray, np.ndarray):
    x = np.load(f'./embeddings/x{path}.npy')
    y = np.load(f'./embeddings/y{path}.npy')
    return x, y
