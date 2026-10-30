import numpy as np
import torch

def get_batch(dataset, batch_size, context_length, device):
    number_of_starts = len(dataset) - context_length
    starts = np.random.randint(
        low=0,
        high=number_of_starts,
        size=batch_size,
    )
    offsets = np.arange(context_length)
    input_indices = starts[:, None] + offsets
    x_array = dataset[input_indices]
    y_array = dataset[input_indices + 1]
    x = torch.tensor(
        x_array,
        dtype=torch.long,
        device=device,
    ) 
    y = torch.tensor(
        y_array,
        dtype=torch.long,
        device=device,
    )
    return x, y