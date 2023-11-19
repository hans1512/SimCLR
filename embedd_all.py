import numpy as np
from src.model import ContrastiveEncoder
from utils.load_data import Loader
from utils.arguments import get_config, get_arguments
from utils.utils import path_from_config

def transform_data(data_loader, check_samples=False):
    # Training dataset
    train_loader = data_loader.train_loader
    # Validation dataset
    test_loader = data_loader.test_loader
    # Show samples from training set
    # Get training and test data. Iterator returns a tuple of 3 variables. Pick the first ones as Xtrain, and Xtest
    ((Xtrain, _), ytrain) = next(iter(train_loader))
    ((Xtest, _), ytest)  = next(iter(test_loader))
    # Print informative message as a sanity check
    print(f"Number of samples in training set: {Xtrain.shape}")
    # Make it a 2D array of batch_size x remaining dimension so that we can use it with PCA for baseline performance
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
        # Also save the original images for plotting
        np.save(f'./embeddings/original_images{save_path}.npy', original_images.numpy())
        np.save(f'./embeddings/original_labels{save_path}.npy', original_labels.numpy())

    return x, y, original_images, original_labels

if __name__ == "__main__":
    config = get_config(get_arguments())
    config["batch_size"] = 1000
    embed(config)