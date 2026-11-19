# Tiny Language Model
A project made to learn more about machine learning, specifically how LLMs work, by training a small text model from scratch in PyTorch.
The dataset used for training is (a small portion of) the huge [Hugging Face Wikipedia dataset](https://huggingface.co/datasets/wikimedia/wikipedia)

## Cool-ish features
  -  Custom per-character tokenizer
  -  Very simple training loop
  -  Simple tests to see how the model behaves

## Showcase
<img width="971" height="179" alt="Model completing a paragraph that was in its training data perfectly" src="https://github.com/user-attachments/assets/b347bc93-ba8c-45ab-b24c-90bcef147ce4" />
<img width="970" height="90" alt="Model failing horribly at trying to complete a sentence not in its training dataset" src="https://github.com/user-attachments/assets/2682f154-ca3b-4dc6-b136-7da6a43be003" />

<i>An example of an early working version of this model, dangerously overfitted ;(</i>


## Trying it yourself
Dependencies:
  -  [PyTorch](https://pytorch.org/)
  -  [numpy](https://numpy.org/)
  -  [pandas](https://pandas.pydata.org/)
  -  [datasets](https://huggingface.co/docs/datasets)

Quick install:
```bash
pip install torch numpy pandas datasets
```
<i>If you want GPU support for PyTorch, grab the right install command for your setup from the [PyTorch install page](https://pytorch.org/get-started/locally/) instead.</i>

To run the project:
```bash
python main.py
```

You will be prompted if you want to train the model or load it. After that, congrats! You have your very own dumb and unoptimized tiny language model!
