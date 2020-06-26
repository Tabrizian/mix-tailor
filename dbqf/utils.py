import torch


def flatten_model(gradient):
    flatt_params = []
    for layer_parameters in gradient:
        flatt_params.append(torch.flatten(layer_parameters))

    return torch.cat(flatt_params)


def flatten_grad(model):
    # flatten the model gradients
    flatt_params = []
    for params in model:
        flatt_params.append(torch.flatten(params))

    return torch.cat(flatt_params)


def unflatten(gradient, parameters, tensor=False):
    """Change the shape of the gradient to the shape of the parameters
        Parameters:
            gradient: flattened gradient
            parameters: convert the flattened gradient to the unflattened
                        version
            tensor: convert to tonsor otherwise it will be an array
    """
    shaped_gradient = []
    begin = 0
    for layer in parameters:
        size = layer.view(-1).shape[0]
        shaped_gradient.append(
            gradient[begin:begin+size].view(layer.shape))
        begin += size
    if tensor:
        return torch.stack(shaped_gradient)
    else:
        return shaped_gradient
