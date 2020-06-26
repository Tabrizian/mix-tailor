from collections import OrderedDict


def mnist_krum(args):
    dataset = 'mnist'
    log_dir = 'runs_rebuttal_mnist_sgd'
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
    stochastic_aggs = ['comed,krum,geomed,bulyan']
    #stochastic_aggs_comp = ['comed,krum,geomed', 'comed,geomed,bulyan', 'comed,krum,bulyan', 'krum,geomed,bulyan']

    args_bsgd_attack = [
        ('world-size', [12]),
        ('distributed', ''),
        ('aggregator', [
            'omniscient',
            'krum',
            ('stochastic', OrderedDict([
                ('stochastic-aggs', stochastic_aggs)
            ]))
        ]),
        ('attack', [
            ('epsilonreverse', OrderedDict([
                ('eps-reverse', [0.2])
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


def mnist_alittle(args):
    dataset = 'mnist'
    log_dir = 'runs_rebuttal_mnist_sgd'
    exclude = ['dataset', 'epochs', 'lr_decay_epoch', 'g_epoch', 'batch_size', 'cuda', 'momentum', 'arch', 'optim', 'agg_grad', 'batch_size']
    shared_args = [('dataset', dataset),
                   ('lr', [0.1, 0.001, 0.01]),  # [.1, .05, .01]),
                   ('weight-decay', 0.0001),
                   ('momentum', 0.9),  # [0, 0.9]),
                   ('niters', 50000),
                   ('workers', 10),  # Data Loaders
                   ('cuda', ''),
                   ('seed', [3]),
                   ('batch-size', [64, 128]),
                   ('arch', ['cnn']),  # 'cnn', 'mlp'
                   ]
    args_sgd = [('optim', ['sgd'])]
    stochastic_aggs = ['comed,krum,geomed,bulyan']
    #stochastic_aggs_comp = ['comed,krum,geomed', 'comed,geomed,bulyan', 'comed,krum,bulyan', 'krum,geomed,bulyan']

    args_bsgd_attack = [
        ('world-size', [12]),
        ('distributed', ''),
        ('aggregator', [
            'omniscient',
            'krum',
            ('stochastic', OrderedDict([
                ('stochastic-aggs', stochastic_aggs)
            ]))
        ]),
        ('attack', [
            ('alittle', OrderedDict([
                ('alittle-num', [1.5])
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

