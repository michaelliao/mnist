#!/usr/bin/env python3

import torch
from torchvision import transforms

from PIL import Image, ImageOps
from model import NeuralNetwork


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using {device}')
model = NeuralNetwork().to(device)
path = './mnist.pth'
model.load_state_dict(torch.load(path))
print(f'loaded model from {path}')
print(model)


def test(path):
    print(f'test {path}...')
    image = Image.open(path).convert('RGB').resize((28, 28))
    image = ImageOps.invert(image)

    trans = transforms.Compose([
        transforms.Grayscale(1),
        transforms.ToTensor()
    ])
    image_tensor = trans(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.nn.functional.softmax(output[0], 0)
    predict = torch.argmax(probs).item()
    return predict, probs[predict], probs


def main():
    for i in range(10):
        predict, prob, probs = test(f'./input/test-{i}.png')
        print(f'expected {i}, actual {predict}, {prob}, {probs}')


if __name__ == '__main__':
    main()
