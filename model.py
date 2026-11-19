import torch
import torch.nn as nn

import torch.nn.functional as F

import model_variables

class TinyLanguageModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


        self.model = nn.Sequential(
            nn.Linear(model_variables.IN_FEATURES * model_variables.VOCABULARY_SIZE, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, model_variables.VOCABULARY_SIZE)
        )


    def forward(self, x):

        missing = model_variables.IN_FEATURES - x.shape[-1]

        if missing > 0:
            x = nn.functional.pad(x, (missing, 0))

        if missing < 0:
            x = x[..., -missing:]

        x = F.one_hot(x.to(torch.long), model_variables.VOCABULARY_SIZE).float()
        x = x.flatten(start_dim=-2)
        
        return self.model(x)

model = TinyLanguageModel()

print("Model parameters:", sum(p.numel() for p in model.parameters()))


def save_model():
    torch.save(model.state_dict(), "model.pth")

def load_model():
    return torch.load("model.pth", weights_only=True)