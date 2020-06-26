import torch
import torch.nn
import dbqf.models.imagenet
import dbqf.models.cifar10
import dbqf.models.loss
import dbqf.models.mnist
from torch.nn.parallel import DistributedDataParallel as DDP


def init_model(opt):
    if opt.dataset == 'mnist':
        if opt.arch == 'cnn':
            model = dbqf.models.mnist.Convnet(not opt.nodropout)
        elif opt.arch == 'bigcnn':
            model = dbqf.models.mnist.BigConvnet(not opt.nodropout)
        elif opt.arch == 'mlp':
            model = dbqf.models.mnist.MLP(not opt.nodropout)
        elif opt.arch == 'smlp':
            model = dbqf.models.mnist.SmallMLP(not opt.nodropout)
        elif opt.arch == 'ssmlp':
            model = dbqf.models.mnist.SuperSmallMLP(not opt.nodropout)
    elif (opt.dataset == 'cifar10' or opt.dataset == 'svhn'
          or opt.dataset == 'cifar100'):
        if opt.arch == 'cnn':
            model = dbqf.models.cifar10.Convnet(num_class=opt.num_class)
        elif opt.arch == 'mlp':
            model = models.cifar10.MLP(num_class=opt.num_class)
        elif opt.arch.startswith('wrn'):
            depth, widen_factor = map(int, opt.arch[3:].split('-'))
            model = models.cifar10_wresnet.WideResNet(
                depth, opt.num_class, widen_factor, 0.3)
        else:
            model = dbqf.models.cifar10.__dict__[opt.arch](
                num_class=opt.num_class)
    elif opt.dataset == 'imagenet':
        model = dbqf.models.imagenet.Model(opt.arch, opt.pretrained)
    elif opt.dataset.startswith('imagenet'):
        model = models.imagenet.Model(opt.arch, opt.pretrained, opt.num_class)
    elif opt.dataset == 'logreg':
        model = models.logreg.Linear(opt.dim, opt.num_class)
    elif opt.dataset == '10class':
        model = models.logreg.Linear(opt.dim, opt.num_class)
    elif opt.dataset == '5class':
        model = models.logreg.Linear(opt.dim, opt.num_class)
    # if opt.distributed:
        # model = DDP(model.cuda(local_rank), device_ids=[local_rank])
    # else:
        # model = torch.nn.DataParallel(model)

    model.criterion = dbqf.models.loss.nll_loss

    return model
