#!/usr/bin/env python3

from time import time

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


from model import NeuralNetwork


def train(dataloader, device, model, loss_fn, optimizer):
    model.train()
    running_loss = 0.0
    for batch, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'loss: {running_loss/len(dataloader):>0.3f}')


def test(dataloader, device, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'accuracy: {100.0*correct/total:>0.2f} %')


def main():
    print('loading training data...')
    train_data = datasets.MNIST(
        root='./data', train=True, download=True, transform=ToTensor())
    print('loading test data...')
    test_data = datasets.MNIST(
        root='./data', train=False, download=True, transform=ToTensor())

    train_dataloader = DataLoader(train_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'using {device}')
    model = NeuralNetwork().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 5
    for t in range(epochs):
        start_time = time()
        print(f'epoch {t+1} / {epochs}\n--------------------')
        train(train_dataloader, device, model, loss_fn, optimizer)
        test(test_dataloader, device, model)
        end_time = time()
        print(f'time: {end_time-start_time:>0.2f} seconds')
    print('done!')
    path = 'mnist.pth'
    torch.save(model.state_dict(), path)
    print(f'model saved: {path}')


if __name__ == '__main__':
    main()
