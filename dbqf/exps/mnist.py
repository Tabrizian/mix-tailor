from collections import OrderedDict


def mnist_simple(args):
    dataset = 'mnist'
    log_dir = 'runs_%s_simple' % dataset
    exclude = ['dataset', 'epochs', 'lr_decay_epoch', 'g_epoch']
    shared_args = [('dataset', dataset),
                   ('lr', [0.01]),  # [.1, .05, .01]),
                   # ('weight-decay', 0.0001),
                   ('momentum', 0),  # [0, 0.9]),
                   ('niters', 3000),
                   ('workers', 0),  # Data Loaders
                   ('cuda', ''),
                   ('batch-size', [32]),
                   ('arch', ['cnn']),  # 'cnn', 'mlp'
                   ]
    args_sgd = [('optim', ['sgd'])]

    args_bsgd = [
        ('heter', ''),
        ('distributed', ''),
        ('aggregator', [
            ('bulyan', OrderedDict([
                ('bul-agg', ['comed']),
                ('bul-sel', ['fedavg']),
            ]))
        ]),
        ('local-rounds', 1),
        ('agg-grad', ''),
        ('num-byz', 2),
        ('attack', ['gaussian', 'reverse', 'mimic'])
    ]

    args += [OrderedDict(shared_args+args_sgd+args_bsgd)]

    return args, log_dir, exclude


def mnist_hetr_attack(args):
    dataset = 'mnist'
    log_dir = 'runs_%s_bsgd' % dataset
    exclude = ['dataset', 'epochs', 'lr_decay_epoch', 'g_epoch']
    shared_args = [('dataset', dataset),
                   ('lr', [0.01]),  # [.1, .05, .01]),
                   # ('weight-decay', 0.0001),
                   ('momentum', 0),  # [0, 0.9]),
                   ('niters', 3000),
                   ('workers', 0),  # Data Loaders
                   ('cuda', ''),
                   ('batch-size', [32]),
                   ('arch', ['cnn']),  # 'cnn', 'mlp'
                   ]
    args_sgd = [('optim', ['sgd'])]

    args_bsgd = [
        ('heter', ''),
        ('distributed', ''),
        ('aggregator', [
            'fedavg',
            'krum',
            'geomed',
            ('bulyanp1', OrderedDict([
                ('bul-agg', ['geomed', 'krum', 'fedavg', 'comed']),
                ('bul-sel', ['geomed', 'krum', 'fedavg', 'comed']),
            ])),
            ('bulyanmkrum', OrderedDict([
                ('bul-agg', ['geomed', 'krum', 'fedavg', 'comed']),
            ])),
            ('bulyan', OrderedDict([
                ('bul-agg', ['geomed', 'krum', 'fedavg', 'comed']),
                ('bul-sel', ['geomed', 'krum', 'fedavg', 'comed']),
            ]))
        ]),
        ('local-rounds', 1),
        ('agg-grad', ''),
        ('num-byz', 2),
        ('attack', ['gaussian', 'reverse', 'mimic'])
    ]

    args += [OrderedDict(shared_args+args_sgd+args_bsgd)]

    return args, log_dir, exclude


def mnist_hetr_woattack(args):
    """
    MNIST heterogenous setting without any attackers.
    """
    dataset = 'mnist'
    log_dir = 'runs_%s_bsgd' % dataset
    exclude = ['dataset', 'epochs', 'lr_decay_epoch', 'g_epoch', 'batch_size', 'cuda', 'momentum', 'arch', 'optim']
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

    args_bsgd = [
        ('heter', ''),
        ('distributed', ''),
        ('aggregator', [
            'fedavg',
            'krum',
            'geomed',
            ('bulyan', OrderedDict([
                ('bul-agg', ['geomed', 'krum'])
            ]))
        ]),
        ('local-rounds', 1),
        ('agg-grad', ''),
        ('num-byz', 2),
    ]

    args += [OrderedDict(shared_args+args_sgd+args_bsgd)]

    return args, log_dir, exclude


def mnist_homo_woattack(args):
    """
    MNIST homogeneous setting without any attackers.
    """
    dataset = 'mnist'
    log_dir = 'runs_%s_homo_sgd' % dataset
    exclude = ['dataset', 'epochs', 'lr_decay_epoch', 'g_epoch']
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

    args_bsgd = [
        ('distributed', ''),
        ('aggregator', [
            'fedavg',
            'krum',
            'geomed',
            ('bulyan', OrderedDict([
                ('bul-agg', ['geomed', 'krum', 'fedavg', 'comed']),
                ('bul-sel', ['geomed', 'krum', 'fedavg', 'comed']),
            ]))
        ]),
        ('local-rounds', 1),
        ('agg-grad', ''),
    ]

    args += [OrderedDict(shared_args+args_sgd+args_bsgd)]

    return args, log_dir, exclude


def mnist_alittle_homo(args):
    """
    MNIST homogeneous setting with a little attack
    """
    dataset = 'mnist'
    log_dir = 'runs_%s_sgd' % dataset
    exclude = ['dataset', 'epochs', 'lr_decay_epoch', 'g_epoch']
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

    args_bsgd = [
        ('distributed', ''),
        ('aggregator', [
            'fedavg',
            'krum',
            'geomed'
        ]),
        ('attack', 'alittle'),
        ('num-byz', 2),
        ('local-rounds', 1),
        ('agg-grad', ''),
    ]

    args += [OrderedDict(shared_args+args_sgd+args_bsgd)]

    return args, log_dir, exclude


def mnist_stochastic(args):
    dataset = 'mnist'
    # log_dir = 'runs_stoch_mnist_sgd'
    log_dir = 'runs_stoch_rebuttal_sgd'
    exclude = ['dataset', 'epochs', 'lr_decay_epoch', 'g_epoch', 'batch_size', 'cuda', 'momentum', 'arch', 'optim', 'agg_grad', 'batch_size']
    shared_args = [('dataset', dataset),
                   ('lr', [0.001]),  # [.1, .05, .01]),
                   ('weight-decay', 0.0001),
                   ('momentum', 0.9),  # [0, 0.9]),
                   ('niters', 50000),
                   ('workers', 10),  # Data Loaders
                   ('cuda', ''),
                   ('seed', [3]),
                   ('batch-size', [50]),
                   ('arch', ['cnn']),  # 'cnn', 'mlp'
                   ]
    args_sgd = [('optim', ['sgd'])]
    # stochastic_aggs = ['comed,krum,geomed,bulyan']
    stochastic_aggs_comp = ['comed,krum,geomed', 'comed,geomed,bulyan', 'comed,krum,bulyan', 'krum,geomed,bulyan']

    args_bsgd_attack = [
        ('world-size', [12]),
        ('distributed', ''),
        ('aggregator', [
            'comed',
            'krum',
            ('stochastic', OrderedDict([
                ('stochastic-aggs', stochastic_aggs_comp)
            ]))
        ]),
        ('attack', [
            ('epsilonreverse', OrderedDict([
                ('eps-reverse', [5.05, 0.5])
                ])),
        ]),
        ('num-byz', [2]),
        ('local-rounds', 1),
        ('agg-grad', ''),
    ]

    args += [OrderedDict(shared_args+args_sgd+args_bsgd_attack)]
    # args_bsgd_attack = [
    #     ('world-size', [10]),
    #     ('distributed', ''),
    #     ('aggregator', [
    #         'omniscient',
    #         'comed',
    #         'krum',
    #         ('stochastic', OrderedDict([
    #             ('stochastic-aggs', stochastic_aggs)
    #         ]))
    #     ]),
    #     ('num-byz', [0]),
    #     ('local-rounds', 1),
    #     ('agg-grad', ''),
    # ]

    # args += [OrderedDict(shared_args+args_sgd+args_bsgd_attack)]
    return args, log_dir, exclude


def mnist_adaptive(args):
    dataset = 'mnist'
    # log_dir = 'runs_stoch_mnist_sgd'
    log_dir = 'runs_stoch_adaptive_sgd'
    exclude = ['dataset', 'epochs', 'lr_decay_epoch', 'g_epoch', 'batch_size', 'cuda', 'momentum', 'arch', 'optim', 'agg_grad', 'batch_size', 'weight_decay', 'niters', 'workers']
    shared_args = [('dataset', dataset),
                   ('lr', [0.001]),  # [.1, .05, .01]),
                   ('weight-decay', 0.0001),
                   ('momentum', 0.9),  # [0, 0.9]),
                   ('niters', 50000),
                   ('workers', 10),  # Data Loaders
                   ('cuda', ''),
                   ('seed', [3]),
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
            ('adaptive', OrderedDict([
                ('adaptive-attack-eps', '0.1,0.5,1,10,100')
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
            'omniscient'
        ]),
        ('num-byz', [0]),
        ('local-rounds', 1),
        ('agg-grad', ''),
    ]

    args += [OrderedDict(shared_args+args_sgd+args_bsgd_attack)]
    return args, log_dir, exclude


def mnist_stochastic_sampling(args):
    dataset = 'mnist'
    log_dir = 'runs_stoch_mnist_sample'
    exclude = ['dataset', 'epochs', 'lr_decay_epoch', 'g_epoch', 'batch_size', 'cuda', 'momentum', 'arch', 'optim', 'agg_grad', 'batch_size', 'agg-grad', 'local-rounds']
    shared_args = [('dataset', dataset),
                   ('lr', [0.001]),  # [.1, .05, .01]),
                   ('weight-decay', 0.0001),
                   ('momentum', 0.9),  # [0, 0.9]),
                   ('niters', 50000),
                   ('workers', 10),  # Data Loaders
                   ('cuda', ''),
                   ('heter', ''),
                   ('resampling', [2,3]),
                   ('seed', [1]),
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
                ('eps-reverse', [0.1])
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


def mnist_sampling_hetr(args):
    dataset = 'mnist'
    log_dir = 'runs_stoch_mnist_sample'
    exclude = ['dataset', 'epochs', 'lr_decay_epoch', 'g_epoch', 'batch_size', 'cuda', 'momentum', 'arch', 'optim', 'agg_grad', 'batch_size']
    shared_args = [('dataset', dataset),
                   ('lr', [0.001]),  # [.1, .05, .01]),
                   ('weight-decay', 0.0001),
                   ('momentum', 0.9),  # [0, 0.9]),
                   ('niters', 50000),
                   ('workers', 1),  # Data Loaders
                   ('cuda', ''),
                   ('seed', [1]),
                   ('batch-size', [50]),
                   ('arch', ['cnn']),  # 'cnn', 'mlp'
                   ]
    args_sgd = [('optim', ['sgd'])]

    args_bsgd_heter = [
        ('world-size', [10]),
        ('distributed', ''),
        ('heter', ''),
        ('aggregator', [
            'omniscient',
            'comed',
            'krum'
        ]),
        ('resampling', [2, 5]),
        ('num-byz', [0]),
        ('local-rounds', 1),
        ('agg-grad', ''),
    ]

    args += [OrderedDict(shared_args+args_sgd+args_bsgd_heter)]

    args_bsgd_homo = [
        ('world-size', [10]),
        ('distributed', ''),
        ('heter', ''),
        ('aggregator', [
            'omniscient',
            'comed',
            'krum'
        ]),
        ('num-byz', [0]),
        ('local-rounds', 1),
        ('agg-grad', ''),
    ]
    args += [OrderedDict(shared_args+args_sgd+args_bsgd_homo)]
    return args, log_dir, exclude


def mnist_stochastic_hetr(args):
    dataset = 'mnist'
    log_dir = 'runs_stoch_mnist_hetr'
    exclude = ['dataset', 'epochs', 'lr_decay_epoch', 'g_epoch', 'batch_size', 'cuda', 'momentum', 'arch', 'optim', 'agg-grad', 'batch_size', 'local-rounds']
    shared_args = [('dataset', dataset),
                   ('lr', [0.1, 0.001]),  # [.1, .05, .01]),
                   ('weight-decay', 0.0001),
                   ('momentum', 0.9),  # [0, 0.9]),
                   ('niters', 50000),
                   ('workers', 10),  # Data Loaders
                   ('cuda', ''),
                   ('seed', [1]),
                   ('heter', ''),
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


def mnist_reverse_eps_homo(args):
    """
    MNIST homogeneous setting with a little attack
    """
    dataset = 'mnist'
    log_dir = 'runs_stochastic_sgd'
    exclude = ['dataset', 'epochs', 'lr_decay_epoch', 'g_epoch', 'batch_size', 'cuda', 'momentum', 'arch', 'optim', 'agg_grad', 'batch_size']
    shared_args = [('dataset', dataset),
                   ('lr', [0.01]),  # [.1, .05, .01]),
                   # ('weight-decay', 0.0001),
                   ('momentum', 0),  # [0, 0.9]),
                   ('niters', 10000),
                   ('workers', 0),  # Data Loaders
                   ('cuda', ''),
                   ('world-size', 12),
                   ('batch-size', [32]),
                   ('arch', ['cnn']),  # 'cnn', 'mlp'
                   ]
    args_sgd = [('optim', ['sgd'])]

    args_bsgd = [
        ('distributed', ''),
        ('aggregator', [
            'krum',
            'comed',
            'omniscient',
            ('stochastic', OrderedDict([
                ('stochastic-num', [2, 5, 10, 16])
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

    args += [OrderedDict(shared_args+args_sgd+args_bsgd)]

    return args, log_dir, exclude


def mnist_homo_attack(args):
    """
    MNIST homogeneous setting with two attackers.
    """
    dataset = 'mnist'
    log_dir = 'runs_%s_homo_bsgd' % dataset
    exclude = ['dataset', 'epochs', 'lr_decay_epoch', 'g_epoch']
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

    args_bsgd = [
        ('distributed', ''),
        ('aggregator', [
            'fedavg',
            'krum',
            'geomed',
            'bulyan', OrderedDict([
                ('bul-agg', ['geomed', 'krum', 'fedavg', 'comed']),
                ('bul-sel', ['geomed', 'krum', 'fedavg', 'comed']),
            ])
        ]),
        ('local-rounds', 1),
        ('agg-grad', ''),
        ('num-byz', 2),
        ('attack', 'gaussian')
    ]

    args += [OrderedDict(shared_args+args_sgd+args_bsgd)]

    return args, log_dir, exclude


def mnist_reverse_comed(args):
    """
    MNIST homogeneous setting with a little attack
    """
    dataset = 'mnist'
    log_dir = 'runs_comed_mnist_sgd'
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
            'comed',
        ]),
        ('attack', [
            ('epsilonreverse', OrderedDict([
                ('eps-reverse', [0.1, 10])
                ])),
        ]),
        ('num-byz', [5]),
        ('local-rounds', 1),
        ('agg-grad', ''),
    ]

    args += [OrderedDict(shared_args+args_sgd+args_bsgd_attack)]

    args_bsgd_attack = [
        ('world-size', 10),
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


def mnist_reverse_krum(args):
    """
    MNIST homogeneous setting with a little attack
    """
    dataset = 'mnist'
    log_dir = 'runs_krum_mnist_sgd'
    exclude = ['dataset', 'epochs', 'lr_decay_epoch', 'g_epoch', 'batch_size', 'cuda', 'momentum', 'arch', 'optim', 'agg_grad', 'batch_size', 'weight_decay']
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


def mnist_reverse_bulyan(args):
    """
    MNIST homogeneous setting with a little attack
    """
    dataset = 'mnist'
    log_dir = 'runs_bulyan_mnist_sgd'
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
            ('bulyan', OrderedDict([
                ('bul-agg', ['geomed', 'krum', 'fedavg', 'comed']),
                ('bul-sel', ['geomed', 'krum', 'fedavg', 'comed']),
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
        ('world-size', 10),
        ('distributed', ''),
        ('aggregator', [
            'omniscient',
            ('bulyan', OrderedDict([
                ('bul-agg', ['geomed', 'krum', 'fedavg', 'comed']),
                ('bul-sel', ['geomed', 'krum', 'fedavg', 'comed']),
            ]))
        ]),
        ('num-byz', [0]),
        ('local-rounds', 1),
        ('agg-grad', ''),
    ]

    args += [OrderedDict(shared_args+args_sgd+args_bsgd_attack)]
    return args, log_dir, exclude
