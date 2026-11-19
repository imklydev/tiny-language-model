should_train_model = True if input("Should train model? (y/N): ") == 'y' else False

from load_dataset import get_raw_data

raw_data = get_raw_data(1)

from tokenizer import tokenize

data = tokenize(raw_data)

from process_data import get_inputs_and_outputs

training_inputs, training_outputs = get_inputs_and_outputs(data)

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.device(device)

training_inputs = torch.tensor(training_inputs, dtype=torch.float32).to(device)
training_outputs = torch.tensor(training_outputs, dtype=torch.float32).to(device)

from model import model, save_model, load_model

model = model.to(device)

from train import train_model

if should_train_model:
    train_model(model, 100, device, training_inputs, training_outputs)
    save_model()
else:
    model.load_state_dict(load_model())

from test import do_simple_tests

do_simple_tests(model, training_inputs)