import numpy as np
import matplotlib.pyplot as pl

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset

import kornia.augmentation as ag

import torchlayers as layers
import torch_optimizer as toptim
from torchinfo import summary
import timm

from sklearn.model_selection import train_test_split

from time import time
from tqdm import tqdm

from utils import Accuracy, DropBlock2D

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(device)


## Data
X = np.load('X_500_processed.npy')
y = np.load('y_500_processed.npy', allow_pickle=True)

X, X_test, y, y_test = train_test_split(X, y, test_size=0.1, stratify=y)
X, X_val, y, y_val = train_test_split(X, y, test_size=0.1, stratify=y)

train_set = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
valid_set = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
test_set = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

input_shape = (1, 1, 32, 32)
num_train = len(train_set)
num_val = len(valid_set)
num_test = len(test_set)

augment = nn.Sequential(
    ag.RandomAffine(degrees=10, shear=0.1, scale=(0.9, 1.1)),
    ag.RandomRotation(20),
)

preprocess = ag.Normalize(mean=torch.tensor([0.1310]), std=torch.tensor([0.3085]))

# augment = False
preprocess = False

train = DataLoader(train_set, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
valid = DataLoader(valid_set, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
test = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)


## 
# model = nn.Sequential(
#     layers.Conv2d(64, 3),
#     layers.ReLU(),
#     layers.Conv2d(64, 3),
#     DropBlock2D(0.3, 3),
#     layers.ReLU(),
#     layers.BatchNorm(),
#     layers.MaxPool(2),

#     layers.Conv(32, 3, padding='same'),
#     layers.ReLU(),
#     layers.Conv(32, 3, padding='same'),
#     DropBlock2D(0.3, 3),
#     layers.ReLU(),
#     layers.BatchNorm(),
#     layers.MaxPool(2),

#     layers.Flatten(),
#     layers.Linear(500),
# )

class Block(nn.Module):
    def __init__(self, num_channels=[32, 32, 64]):
        super().__init__()
        self.conv11 = layers.Conv(num_channels[0], 1)
        self.conv3 = layers.Conv(num_channels[1], 3, padding='same')
        self.conv12 = layers.Conv(num_channels[2], 1)
        self.conv13 = layers.Conv(num_channels[2], 1)
        self.relu = layers.ReLU()
        self.bn1 = layers.BatchNorm()
        self.bn2 = layers.BatchNorm()
        self.bn3 = layers.BatchNorm()

    def forward(self, x):
        z = self.relu(self.bn1(self.conv11(x)))
        z = self.relu(self.bn2(self.conv3(z)))
        z = self.bn3(self.conv12(z)) + self.conv13(x)
        z = self.relu(z)
        return z


model = nn.Sequential(
    layers.Conv(64, 3, padding='same'),
    layers.ReLU(),
    layers.BatchNorm(),
    layers.MaxPool(2),

    Block([32, 32, 64]),
    layers.MaxPool(2),

    Block([16, 16, 32]),
    DropBlock2D(0.3, 3),
    layers.MaxPool(2),

    layers.Flatten(),
    layers.Linear(500)
)

# model = resnet18()
# model.conv1 = layers.Conv2d(64, 3)
# model.classifier = layers.Linear(500)

model(train_set[0][0][None,:])
model.cuda()
print(summary(model, input_size=input_shape, depth=2))

loss_fn = nn.CrossEntropyLoss()

_optimizer = toptim.AdaBelief(model.parameters(), lr=0.001, weight_decay=5e-5)
optimizer = toptim.Lookahead(_optimizer)
scheduler = optim.lr_scheduler.OneCycleLR(_optimizer, max_lr=0.01, steps_per_epoch=len(train), epochs=100)


##
for epoch in range(100):
    loss_train = 0
    loss_val = 0
    acc = Accuracy(model, num_train, num_val, topk=5)

    t = time()
    model.train()
    for x, y in tqdm(train):
        x, y = x.cuda(), y.cuda()
        if augment: x = augment(x)
        if preprocess: x = preprocess(x)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        try: scheduler.step()
        except: pass

        loss_train += loss.item()
        acc.update(y_pred, y)

    model.eval()
    with torch.no_grad():
        for x, y in valid:
            x, y = x.cuda(), y.cuda()
            if preprocess: x = preprocess(x)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            loss_val += loss.item()
            acc.update(y_pred, y)
    t = time() - t

    loss_train, loss_val = loss_train/num_train, loss_val/num_val

    results = " -> [Epoch {} in {:.1f}s] [Loss: {:.6e} {:.6e}] [Acc: {}]".format(epoch, t, loss_train, loss_val, acc)
    print(results)


##
def test_fn():
    acc = Accuracy(model, num_train, num_test, topk=5)
    model.eval()
    with torch.no_grad():
        for x, y in test:
            x, y = x.cuda(), y.cuda()
            if preprocess: x = preprocess(x)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)

            acc.update(y_pred, y)

    return acc

test_fn()

