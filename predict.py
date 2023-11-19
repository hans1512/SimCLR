import numpy as np
import torch

from classification_head import ClassificationHead
from tqdm import tqdm

from utils.arguments import get_config, get_arguments
from utils.utils import path_from_config


def load(path: str) -> (np.ndarray, np.ndarray):
    x = np.load(f'./embeddings/x{path}.npy')
    y = np.load(f'./embeddings/y{path}.npy')
    return x, y


def predict(model, x, y):
    model.to('cuda')

    predictions = list()
    with torch.no_grad():
        for i in tqdm(range(len(x))):
            _x = torch.tensor(x[i]).to("cuda")
            prediction = model(_x)
            predictions.append(np.argmax(prediction.cpu().detach().numpy()))

    correct = 0
    correct += sum(predictions[i] == y[i] for i in range(len(predictions)))

    print(correct)
    print(f"Accuracy: {correct / len(predictions)}")


def start_prediction():
    config = get_config(get_arguments())
    save_path = path_from_config(config)

    x, y = load(save_path)
    model = ClassificationHead(config)

    model.load_state_dict(torch.load(f"./models/classification_head{save_path}.pth"))

    predict(model, x, y)

start_prediction()
