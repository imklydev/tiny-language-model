from datasets import load_dataset
import pandas as pd


def get_raw_data(num_samples):
    """
    Get wikipedia training data using the datasets library
    """


    dataset = load_dataset("wikimedia/wikipedia", "20231101.en",
                            split="train", cache_dir="D:/datasets", streaming=True)

    data = pd.DataFrame(dataset.take(num_samples))["text"]

    return data