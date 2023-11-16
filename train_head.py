from classification_head import ClassificationHead
from load_embeddings import load
from utils.arguments import get_config, get_arguments
from utils.utils import path_from_config

if __name__ == "__main__":
    config = get_config(get_arguments())
    save_path = path_from_config(config)

    x, y = load(save_path)

    net = ClassificationHead(config)
    net.to('cuda')

    net.fit(x, y)

    net.save()