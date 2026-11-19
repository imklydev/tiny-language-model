import model_variables

def get_inputs_and_outputs(data):
    """
    Split data into 2 lists: inputs (with IN_FEATURES length for each item) and outputs (with one token for every item)
    """


    inputs = []
    outputs = []

    for i in range(0, len(data) - model_variables.IN_FEATURES - 1):
        inputs.append(data[i : i + model_variables.IN_FEATURES])

        next_token = data[i + model_variables.IN_FEATURES]

        token_list = [0 for i in range(model_variables.VOCABULARY_SIZE)]

        token_list[next_token] = 1

        outputs.append(token_list)

    return inputs, outputs