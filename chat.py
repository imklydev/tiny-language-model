from model import model
import torch

from main import device

def continue_sentence(sentence, chars_to_predict, include_input_sentence=True):


    inputs = sentence

    predictions = []

    while(chars_to_predict):

        prediction = torch.argmax(model(inputs)).to(device)

        inputs = inputs.tolist()

        inputs = inputs[1:]
        inputs.append(prediction)


        predictions.append(prediction)

        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)

        chars_to_predict -= 1

    if(include_input_sentence):
        return torch.tensor(sentence.tolist() + predictions)
    else:
        return torch.tensor(predictions)