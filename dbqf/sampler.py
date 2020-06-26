import torch

from dbqf.dataloaders import get_loaders


class Sampler:

    def __init__(self, data_loader, logger, opt):
        self.opt = opt
        self.dataiter = iter(data_loader)

    def sample(self, model):
        dt = self.dataiter
        model.zero_grad()
        data = next(dt)
        loss = model.criterion(model, data)
        grad = torch.autograd.grad(loss, model.parameters())
        return grad

    def get_mean(self, model, num_samples):
        pass

    def get_var(self, model, num_samples):
        pass
