import ipdb
import torch
import random
import torch.nn.functional as F

from dbqf.aggregators import get_aggregator
from dbqf.attacks import get_attack
from dbqf.dataloaders import get_loaders, InfiniteLoader
from dbqf.utils import flatten_model
from dbqf.utils import flatten_grad, unflatten


class OptimizerFactory(object):

    def __init__(self, model, logger, opt, world_size):
        self.model = model
        self.opt = opt
        # current iterations
        self.niters = 0
        self.optimizer = None
        # current epoch
        self.epoch = 0
        self.logger = logger
        train_loaders = []
        self.resampling = opt.resampling
        self.num_byz = opt.num_byz

        # The batch size should be divided when doing resampling, otherwise
        # the batch size should not be changed.
        if opt.heter:
            for i in range(world_size):
                train_loader, _, _ = get_loaders(opt, i)
                train_loaders.append(InfiniteLoader(train_loader))
        else:
            train_loader, _, _ = get_loaders(opt)
            train_loaders.append(InfiniteLoader(train_loader))
        self.train_loaders = train_loaders
        self.world_size = world_size
        self.reset()

        datasets = []
        for train_loader in train_loaders:
            datasets.append(iter(train_loader))
        self.datasets = datasets

        grad_size = len(flatten_model(model.parameters()))
        self.attack = get_attack(opt, grad_size)
        self.aggregator = get_aggregator(opt, self, logger)

    def log_grad(self, model, tag):
        logger = self.logger
        param_tolog = list(model.parameters())[0].grad.view(-1)[0:5]
        for i, param in enumerate(param_tolog):
            logger.log_value(tag+str(i), param, self.niters)

    def log_param(self, model, tag):
        logger = self.logger
        param_tolog = list(model.parameters())[0].view(-1)[0:5]
        for i, param in enumerate(param_tolog):
            logger.log_value(tag+str(i), param, self.niters)

    def reset(self):
        model = self.model
        opt = self.opt
        if opt.optim == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=opt.lr, momentum=opt.momentum,
                                        weight_decay=opt.weight_decay,
                                        nesterov=opt.nesterov)
        else:
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=opt.lr,
                                         weight_decay=opt.weight_decay)
        self.epoch = 0
        self.niters = 0
        self.optimizer = optimizer

    def adjust_learning_rate(self):
        pass

    def get_sample(self, worker_index):
        len_datasets = len(self.datasets)
        input, target, _ = next(self.datasets[worker_index % len_datasets])
        return input, target

    def step(self):
        opt = self.opt
        model = self.model
        optimizer = self.optimizer
        world_size = self.world_size
        resampling = self.resampling

        grads = []
        full_loss = 0

        num_nonbyz = self.world_size - self.num_byz
        for i in range(num_nonbyz):
            input, target = self.get_sample(i)
            input = input.to('cuda')
            target = target.to('cuda')
            self.optimizer.zero_grad()
            model.zero_grad()
            output = model(input)
            loss = F.nll_loss(output, target)
            full_loss += loss
            grad = torch.autograd.grad(loss, model.parameters())

            grads.append(flatten_grad(grad))

        attack_grads = []
        for i in range(num_nonbyz, world_size):
            attack_grads.append(self.attack.grad(torch.stack(grads)))

        grads += attack_grads

        new_grads = []
        for i in range(world_size):
            new_grads.append(torch.zeros_like(grads[i]))
        if resampling:
            item_list = list(range(len(grads)))
            total_list = []
            for j in range(resampling):
                total_list += item_list
            total_list = torch.tensor(total_list)[torch.randperm(len(total_list))]

            # Implementation of resampling according to the paper.
            for i in range(len(total_list)):
                new_grads[i//(resampling)] += grads[total_list[i]] / resampling

            grads = new_grads

        # model weights before applying grads
        if opt.agg_grad:
            agg_grads = self.aggregator.agg_grad(torch.stack(grads))

        params_grad = unflatten(agg_grads, model.parameters())
        for param, grad in zip(model.parameters(), params_grad):
            if param.grad is None:
                param.grad = grad.clone()
            else:
                param.grad.copy_(grad)

        optimizer.step()

        # model weights after applying grads
        self.log_param(model, 'model_ag/')

        if not opt.agg_grad:
            # model weights after averaging
            self.aggregator.agg_weight(model)

        self.log_param(model, 'model_aa/')
        self.niters += 1
        return full_loss / len(grads)
