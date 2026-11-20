import random

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from model_variables import IN_FEATURES

from tokenizer import PAD_ID

def train_model(model, epochs, device, training_inputs, training_outputs):

    dataset = TensorDataset(training_inputs, training_outputs)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)

    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters())

    model.train()
    for epoch in range(epochs):

        positions = torch.arange(IN_FEATURES, device=device).unsqueeze(0)

        losses = []
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            # number of leading tokens to PAD out, per sample
            k = torch.randint(0, IN_FEATURES, (X.shape[0], 1), device=device)
            X = X.masked_fill(positions < k, PAD_ID)

            prediction = model(X)

            loss = loss_fn(prediction, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.detach())

        if (epoch + 1) % 1 == 0:
            print(f"Epoch {epoch + 1}: loss = {sum(losses) / len(losses)}")
