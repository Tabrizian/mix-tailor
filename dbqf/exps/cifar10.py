from collections import OrderedDict


def cifar10(args):
    dataset = 'cifar10'
    log_dir = 'runs_%s_bsgd' % dataset
    exclude = ['dataset', 'epochs', 'lr_decay_epoch', 'g_epoch']
    shared_args = [('dataset', dataset),
                   ('lr', .1),  # [.1, .05, .01]),
                   ('weight-decay', 0.0001),
                   ('momentum', 0.9),  # [0, 0.9]),
                   ('niters', 80000),
                   ('workers', 10),  # Data Loaders
                   ('cuda', ''),
                   ('arch', ['resnet20']),  # 'cnn', 'mlp'
                   ]
    args_sgd = [('optim', ['sgd'])]

    args_bsgd = [
        ('heter', ''),
        ('distributed', ''),
        ('aggregator', ['fedavg']),
        ('local-rounds', 5),
    ]

    args += [OrderedDict(shared_args+args_sgd+args_bsgd)]

    return args, log_dir, exclude


def cifar10_stochastic_sampling(args):
    dataset = 'cifar10'
    log_dir = 'runs_stoch_cifar10_sampling'
    exclude = ['dataset', 'epochs', 'lr_decay_epoch', 'g_epoch', 'batch_size', 'cuda', 'momentum', 'arch', 'optim', 'agg-grad', 'batch_size', 'local-rounds']
    shared_args = [('dataset', dataset),
                   ('lr', [0.001]),  # [.1, .05, .01]),
                   ('weight-decay', 0.0001),
                   ('momentum', 0.9),  # [0, 0.9]),
                   ('niters', 50000),
                   ('workers', 10),  # Data Loaders
                   ('cuda', ''),
                   ('seed', [1]),
                   ('heter', ''),
                   ('resampling', [2, 3]),
                   ('batch-size', [50]),
                   ('arch', ['cnn']),  # 'cnn', 'mlp'
                   ]
    args_sgd = [('optim', ['sgd'])]
    stochastic_aggs = ['comed,krum,geomed,bulyan']

    args_bsgd_attack = [
        ('world-size', [12]),
        ('distributed', ''),
        ('aggregator', [
            'comed',
            'krum',
            ('stochastic', OrderedDict([
                ('stochastic-aggs', stochastic_aggs)
            ]))
        ]),
        ('attack', [
            ('epsilonreverse', OrderedDict([
                ('eps-reverse', [0.1, 10])
                ])),
        ]),
        ('num-byz', [2]),
        ('local-rounds', 1),
        ('agg-grad', ''),
    ]

    args += [OrderedDict(shared_args+args_sgd+args_bsgd_attack)]
    args_bsgd_attack = [
        ('world-size', [10]),
        ('distributed', ''),
        ('aggregator', [
            'omniscient',
            'comed',
            'krum',
            ('stochastic', OrderedDict([
                ('stochastic-aggs', stochastic_aggs)
            ]))
        ]),
        ('num-byz', [0]),
        ('local-rounds', 1),
        ('agg-grad', ''),
    ]

    args += [OrderedDict(shared_args+args_sgd+args_bsgd_attack)]
    return args, log_dir, exclude


def cifar10_stochastic(args):
    dataset = 'cifar10'
    log_dir = 'runs_stoch_cifar10_sgd'
    exclude = ['dataset', 'epochs', 'lr_decay_epoch', 'g_epoch', 'batch_size', 'cuda', 'momentum', 'arch', 'optim', 'agg_grad', 'batch_size']
    shared_args = [('dataset', dataset),
                   ('lr', [0.1, 0.001]),  # [.1, .05, .01]),
                   ('weight-decay', 0.0001),
                   ('momentum', 0.9),  # [0, 0.9]),
                   ('niters', 50000),
                   ('workers', 10),  # Data Loaders
                   ('cuda', ''),
                   ('seed', [1]),
                   ('batch-size', [50, 80]),
                   ('arch', ['cnn']),  # 'cnn', 'mlp'
                   ]
    args_sgd = [('optim', ['sgd'])]
    stochastic_aggs = ['comed,krum,geomed,bulyan']

    args_bsgd_attack = [
        ('world-size', [12]),
        ('distributed', ''),
        ('aggregator', [
            'comed',
            'krum',
            ('stochastic', OrderedDict([
                ('stochastic-aggs', stochastic_aggs)
            ])),
            ('bulyan', OrderedDict([
                ('bul-agg', ['comed']),
                ('bul-sel', ['krum']),
            ])),
            'geomed'
        ]),
        ('attack', [
            ('epsilonreverse', OrderedDict([
                ('eps-reverse', [0.1, 10])
                ])),
        ]),
        ('num-byz', [2]),
        ('local-rounds', 1),
        ('agg-grad', ''),
    ]

    args += [OrderedDict(shared_args+args_sgd+args_bsgd_attack)]
    args_bsgd_attack = [
        ('world-size', [10]),
        ('distributed', ''),
        ('aggregator', [
            'omniscient',
            'comed',
            'krum',
            ('stochastic', OrderedDict([
                ('stochastic-aggs', stochastic_aggs)
            ]))
        ]),
        ('num-byz', [0]),
        ('local-rounds', 1),
        ('agg-grad', ''),
    ]

    args += [OrderedDict(shared_args+args_sgd+args_bsgd_attack)]
    return args, log_dir, exclude


def cifar10_reverse_comed(args):
    """
    MNIST homogeneous setting with a little attack
    """
    dataset = 'cifar10'
    log_dir = 'runs_comed_cifar10_sgd'
    exclude = ['dataset', 'epochs', 'lr_decay_epoch', 'g_epoch', 'batch_size', 'cuda', 'momentum', 'arch', 'optim', 'agg_grad', 'batch_size']
    shared_args = [('dataset', dataset),
                   ('lr', [0.1]),  # [.1, .05, .01]),
                   ('weight-decay', 0.0001),
                   ('momentum', 0.9),  # [0, 0.9]),
                   ('niters', 50000),
                   ('workers', 10),  # Data Loaders
                   ('cuda', ''),
                   ('seed', [1]),
                   ('batch-size', [50]),
                   ('arch', ['cnn']),  # 'cnn', 'mlp'
                   ]
    args_sgd = [('optim', ['sgd'])]

    # args_bsgd_attack = [

    #     ('world-size', 25),
    #     ('distributed', ''),
    #     ('aggregator', [
    #         'comed',
    #     ]),
    #     ('attack', [
    #         ('epsilonreverse', OrderedDict([
    #             ('eps-reverse', [0.1, 10])
    #             ])),
    #     ]),
    #     ('num-byz', [11]),
    #     ('local-rounds', 1),
    #     ('agg-grad', ''),
    # ]

    # args += [OrderedDict(shared_args+args_sgd+args_bsgd_attack)]

    args_bsgd_attack = [
        ('world-size', [13]),
        ('distributed', ''),
        ('aggregator', [
            'omniscient',
            'comed',
        ]),
        ('num-byz', [0]),
        ('local-rounds', 1),
        ('agg-grad', ''),
    ]

    args += [OrderedDict(shared_args+args_sgd+args_bsgd_attack)]
    return args, log_dir, exclude


def cifar10_reverse_krum(args):
    """
    MNIST homogeneous setting with a little attack
    """
    dataset = 'cifar10'
    log_dir = 'runs_krum_cifar10_sgd'
    exclude = ['dataset', 'epochs', 'lr_decay_epoch', 'g_epoch', 'batch_size', 'cuda', 'momentum', 'arch', 'optim', 'agg_grad', 'batch_size']
    shared_args = [('dataset', dataset),
                   ('lr', [0.01]),  # [.1, .05, .01]),
                   # ('weight-decay', 0.0001),
                   ('momentum', 0),  # [0, 0.9]),
                   ('niters', 10000),
                   ('workers', 0),  # Data Loaders
                   ('cuda', ''),
                   ('batch-size', [32]),
                   ('arch', ['cnn']),  # 'cnn', 'mlp'
                   ]
    args_sgd = [('optim', ['sgd'])]

    args_bsgd_attack = [
        ('world-size', 12),
        ('distributed', ''),
        ('aggregator', [
            'krum',
        ]),
        ('attack', [
            ('epsilonreverse', OrderedDict([
                ('eps-reverse', [0.1, 10])
                ])),
        ]),
        ('num-byz', [2]),
        ('local-rounds', 1),
        ('agg-grad', ''),
    ]

    args += [OrderedDict(shared_args+args_sgd+args_bsgd_attack)]

    args_bsgd_attack = [
        ('world-size', 10),
        ('distributed', ''),
        ('aggregator', [
            'omniscient',
            'krum',
        ]),
        ('num-byz', [0]),
        ('local-rounds', 1),
        ('agg-grad', ''),
    ]

    args += [OrderedDict(shared_args+args_sgd+args_bsgd_attack)]
    return args, log_dir, exclude


def cifar10_stochastic_hetr(args):
    dataset = 'cifar10'
    log_dir = 'runs_stoch_cifar10_hetr'
    exclude = ['dataset', 'epochs', 'lr_decay_epoch', 'g_epoch', 'batch_size', 'cuda', 'momentum', 'arch', 'optim', 'agg_grad', 'batch_size']
    shared_args = [('dataset', dataset),
                   ('lr', [0.1, 0.001]),  # [.1, .05, .01]),
                   ('weight-decay', 0.0001),
                   ('momentum', 0.9),  # [0, 0.9]),
                   ('niters', 50000),
                   ('workers', 10),  # Data Loaders
                   ('cuda', ''),
                   ('heter', ''),
                   ('seed', [1]),
                   ('batch-size', [50]),
                   ('arch', ['cnn']),  # 'cnn', 'mlp'
                   ]
    args_sgd = [('optim', ['sgd'])]
    stochastic_aggs = ['comed,krum', 'comed,geomed', 'comed,bulyan', 'krum,geomed', 'krum,bulyan', 'geomed,bulyan', 'comed,krum,geomed', 'comed,krum,bulyan', 'comed,geomed,bulyan', 'krum,geomed,bulyan', 'comed,krum,geomed,bulyan']

    args_bsgd_attack = [
        ('world-size', [12]),
        ('distributed', ''),
        ('aggregator', [
            'comed',
            'krum',
            ('stochastic', OrderedDict([
                ('stochastic-aggs', stochastic_aggs)
            ]))
        ]),
        ('attack', [
            ('epsilonreverse', OrderedDict([
                ('eps-reverse', [0.1, 10])
                ])),
        ]),
        ('num-byz', [2]),
        ('local-rounds', 1),
        ('agg-grad', ''),
    ]

    args += [OrderedDict(shared_args+args_sgd+args_bsgd_attack)]
    args_bsgd_attack = [
        ('world-size', [10]),
        ('distributed', ''),
        ('aggregator', [
            'omniscient',
            'comed',
            'krum',
            ('stochastic', OrderedDict([
                ('stochastic-aggs', stochastic_aggs)
            ]))
        ]),
        ('num-byz', [0]),
        ('local-rounds', 1),
        ('agg-grad', ''),
    ]

    args += [OrderedDict(shared_args+args_sgd+args_bsgd_attack)]
    return args, log_dir, exclude


def cifar10_reverse_comed(args):
    """
    MNIST homogeneous setting with a little attack
    """
    dataset = 'cifar10'
    log_dir = 'runs_comed_cifar10_sgd'
    exclude = ['dataset', 'epochs', 'lr_decay_epoch', 'g_epoch', 'batch_size', 'cuda', 'momentum', 'arch', 'optim', 'agg_grad', 'batch_size']
    shared_args = [('dataset', dataset),
                   ('lr', [0.1]),  # [.1, .05, .01]),
                   ('weight-decay', 0.0001),
                   ('momentum', 0.9),  # [0, 0.9]),
                   ('niters', 50000),
                   ('workers', 10),  # Data Loaders
                   ('cuda', ''),
                   ('seed', [1]),
                   ('batch-size', [50]),
                   ('arch', ['cnn']),  # 'cnn', 'mlp'
                   ]
    args_sgd = [('optim', ['sgd'])]

    # args_bsgd_attack = [

    #     ('world-size', 25),
    #     ('distributed', ''),
    #     ('aggregator', [
    #         'comed',
    #     ]),
    #     ('attack', [
    #         ('epsilonreverse', OrderedDict([
    #             ('eps-reverse', [0.1, 10])
    #             ])),
    #     ]),
    #     ('num-byz', [11]),
    #     ('local-rounds', 1),
    #     ('agg-grad', ''),
    # ]

    # args += [OrderedDict(shared_args+args_sgd+args_bsgd_attack)]

    args_bsgd_attack = [
        ('world-size', [13]),
        ('distributed', ''),
        ('aggregator', [
            'omniscient',
            'comed',
        ]),
        ('num-byz', [0]),
        ('local-rounds', 1),
        ('agg-grad', ''),
    ]

    args += [OrderedDict(shared_args+args_sgd+args_bsgd_attack)]
    return args, log_dir, exclude


def cifar10_reverse_krum(args):
    """
    MNIST homogeneous setting with a little attack
    """
    dataset = 'cifar10'
    log_dir = 'runs_krum_cifar10_sgd'
    exclude = ['dataset', 'epochs', 'lr_decay_epoch', 'g_epoch', 'batch_size', 'cuda', 'momentum', 'arch', 'optim', 'agg_grad', 'batch_size']
    shared_args = [('dataset', dataset),
                   ('lr', [0.01]),  # [.1, .05, .01]),
                   # ('weight-decay', 0.0001),
                   ('momentum', 0),  # [0, 0.9]),
                   ('niters', 10000),
                   ('workers', 0),  # Data Loaders
                   ('cuda', ''),
                   ('batch-size', [32]),
                   ('arch', ['cnn']),  # 'cnn', 'mlp'
                   ]
    args_sgd = [('optim', ['sgd'])]

    args_bsgd_attack = [
        ('world-size', 12),
        ('distributed', ''),
        ('aggregator', [
            'krum',
        ]),
        ('attack', [
            ('epsilonreverse', OrderedDict([
                ('eps-reverse', [0.1, 10])
                ])),
        ]),
        ('num-byz', [2]),
        ('local-rounds', 1),
        ('agg-grad', ''),
    ]

    args += [OrderedDict(shared_args+args_sgd+args_bsgd_attack)]

    args_bsgd_attack = [
        ('world-size', 10),
        ('distributed', ''),
        ('aggregator', [
            'omniscient',
            'krum',
        ]),
        ('num-byz', [0]),
        ('local-rounds', 1),
        ('agg-grad', ''),
    ]

    args += [OrderedDict(shared_args+args_sgd+args_bsgd_attack)]
    return args, log_dir, exclude
