import numpy as np
from src.model import ContrastiveEncoder
from utils.load_data import Loader
from utils.arguments import get_config, get_arguments
from utils.utils import path_from_config

def transform_data(data_loader):
    train_loader = data_loader.train_loader
    test_loader = data_loader.test_loader
    ((Xtrain, _), ytrain) = next(iter(train_loader))
    ((Xtest, _), ytest)  = next(iter(test_loader))
    print(f"Number of samples in training set: {Xtrain.shape}")
    Xtrain2D, Xtest2D = Xtrain.view(Xtrain.shape[0], -1), Xtest.view(Xtest.shape[0], -1)
    # Return arrays
    return Xtrain2D, ytrain, Xtest2D, ytest


def embed(config, save=True):
    dataloader = Loader(config, train=False)
    encoder = ContrastiveEncoder(config)
    encoder.load_models()

    # Get original images and their labels
    ((original_images, _), original_labels) = next(iter(dataloader.train_loader))

    # Get embeddings and labels from the encoder
    x, y = encoder.predict(dataloader.train_loader)
    x = np.array(x)
    y = np.array(y)

    if save:
        save_path = path_from_config(config)
        np.save(f'./embeddings/x{save_path}.npy', x)
        np.save(f'./embeddings/y{save_path}.npy', y)
        np.save(f'./embeddings/original_images{save_path}.npy', original_images.numpy())
        np.save(f'./embeddings/original_labels{save_path}.npy', original_labels.numpy())

    return x, y, original_images, original_labels

if __name__ == "__main__":
    config = get_config(get_arguments())
    config["batch_size"] = 1000
    embed(config)