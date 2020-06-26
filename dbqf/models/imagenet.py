import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from torch.nn.parallel import DistributedDataParallel as DDP


class Model(nn.Module):
    def __init__(self, arch, pretrained=False, nclass=None):
        super(Model, self).__init__()
        model = torchvision.models.__dict__[arch](pretrained)
        if nclass is not None and nclass != model.module.fc.out_features:
            model.module.fc = nn.Linear(model.module.fc.in_features,
                                        nclass)
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=-1)
