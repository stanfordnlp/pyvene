import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sigmoid_boundary_sigmoid(_input, boundary_x, boundary_y, temperature):
    return torch.sigmoid((_input - boundary_x) / temperature) * \
        torch.sigmoid((boundary_y - _input) / temperature)


def harmonic_boundary_sigmoid(_input, boundary_x, boundary_y, temperature):
    return (_input<=boundary_x)*torch.sigmoid((_input - boundary_x) / temperature) + \
    (_input>=boundary_y)*torch.sigmoid((boundary_y - _input) / temperature) + \
    ((_input>boundary_x)&(_input<boundary_y))*torch.sigmoid(
        (0.5 * (torch.abs(_input - boundary_x)**(-1) + torch.abs(_input - boundary_y)**(-1)))**(-1) / temperature
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
