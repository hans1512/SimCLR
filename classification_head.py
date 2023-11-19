import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.utils import path_from_config

class ClassificationHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.lin1 = nn.Linear(512, 4096)
        self.lin5 = nn.Linear(4096, 10)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.lin1(x))
        return self.sigmoid(self.lin5(x))

    def index_to_tensor(self, y: int):
        t = torch.zeros(10, dtype=torch.float32)
        t[y] = 1
        return t

    def fit(self, _x: np.ndarray, _y: np.ndarray):
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        for epoch in tqdm(range(10)):
            running_loss = 0.0
            for i, (x_val, y_val) in enumerate(zip(_x, _y)):
                x = torch.tensor(x_val).to('cuda')
                y = self.index_to_tensor(y_val).to('cuda')

                optimizer.zero_grad()
                out = self(x)
                loss = criterion(out, y)
                loss.backward()
                running_loss += loss.item()

                if i % 16 == 15:
                    optimizer.step()

                if i % 2000 == 1999:
                    print(f'Epoch {epoch + 1}, Step {i + 1}, Loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0

            self.save()

    def save(self):
        model_path = f"./models/classification_head{path_from_config(self.config)}.pth"
        torch.save(self.state_dict(), model_path)
        print(f"Model saved to {model_path}")

