import torch

def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }
    torch.save(checkpoint, out)

def load_checkpoint(src, model, optimizer):
    parameter = next(model.parameters(), None)

    if parameter is not None:
        map_location = "cpu"
    else:
        map_location = parameter.device

    checkpoint = torch.load(
        src,
        map_location =map_location,
    )
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]