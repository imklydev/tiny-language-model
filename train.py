
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def train_model(model, epochs, device, training_inputs, training_outputs):

    dataset = TensorDataset(training_inputs, training_outputs)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters())

    model.train()
    for epoch in range(epochs):

        losses = []
        for X, y in dataloader:
            prediction = model(X)

            loss = loss_fn(prediction, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.detach())

        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch + 1}: loss = {sum(losses) / len(losses)}")
